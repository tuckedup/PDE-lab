"""Metric definitions, baseline forecasters and content-identity primitives.

The metrics are **declared before evaluation** (see
``configs/targets.yaml``) and are computed identically for every
compared condition.

nRMSE (primary data metric)
----------------------------
For a held-out trajectory ``i`` and a set of predicted future snapshots
``P_i`` against reference snapshots ``R_i``:

    nRMSE_i = sqrt( sum((P_i - R_i)^2) / (sum(R_i^2) + eps) )

``eps`` is a fixed, declared constant (default 1e-12) that guards
against a zero denominator. Behaviour for near-zero references is
reported explicitly: trajectories whose reference energy is below
``near_zero_ref_threshold`` are flagged and reported separately rather
than being silently included.

The *copied initial frame* is **not** included in the error sum, so
copying the input cannot deflate the metric.

Physics-residual error
-----------------------
The residual error is the physical-unit RMS of the PDE residual over the
declared space/time domain, computed with an identical evaluator,
grid, time spacing and aggregation for the baseline and the proposed
model. The reference solution's own residual under the same evaluator is
reported alongside, so that operator-specific overfitting is visible.

Content identity
----------------
Every number this module produces is bound to a *content identity*: the
SHA-256 of the bytes that produced it. The primitives below compute
those identities (files, canonical JSON, the evaluator implementation,
the normalisation constants) and write artifacts atomically, so that a
reader can tell whether a stored number still corresponds to the code
and the data that are on disk right now.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

EPSILON = 1e-12
NEAR_ZERO_REF_THRESHOLD = 1e-8

#: Version of the metric implementation.  Bumping it invalidates every
#: artifact that was produced with the previous definitions.
METRIC_IMPLEMENTATION_VERSION = "pdelab-metrics/2"

#: The evaluator implementation files whose *bytes* define the identity
#: of an evaluation.  ``physics/operators.py`` is included because the
#: residual evaluators call the spatial operators from there: a change
#: there changes the numbers.
EVALUATOR_IMPLEMENTATION_FILES = (
    "src/pdelab/evaluation/evaluate.py",
    "src/pdelab/evaluation/metrics.py",
    "src/pdelab/evaluation/rollout.py",
    "src/pdelab/physics/operators.py",
)


# ----------------------------------------------------------------------
# content identity
# ----------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    """SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's *finished bytes*."""
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """SHA-256 of a canonical JSON rendering of ``payload``.

    Key order and insignificant whitespace are removed, so two payloads
    that mean the same thing hash the same.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(blob.encode("utf-8"))


def write_bytes_atomic(path: str | Path, data: bytes) -> str:
    """Write ``data`` to ``path`` atomically and return its SHA-256.

    The file is written to a temporary name in the *same* directory,
    flushed, ``fsync``-ed and then ``os.replace``-d over the target, so
    a reader never observes a partially written artifact.  The returned
    digest is computed from the finished bytes *after* the rename.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return sha256_bytes(data)


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> str:
    """Write ``payload`` as JSON atomically; return the SHA-256 of the file.

    A companion ``<name>.sha256`` sidecar is written with the same
    digest, so that a later reader can detect a corrupted or truncated
    artifact instead of silently parsing a partial file.
    """
    p = Path(path)
    blob = (json.dumps(payload, indent=indent, default=str) + "\n").encode("utf-8")
    digest = write_bytes_atomic(p, blob)
    write_bytes_atomic(p.with_suffix(p.suffix + ".sha256"), (digest + "\n").encode("utf-8"))
    return digest


def write_text_atomic(path: str | Path, text: str) -> str:
    """Write text atomically; return the SHA-256 of the finished bytes."""
    return write_bytes_atomic(path, text.encode("utf-8"))


def read_digest_sidecar(path: str | Path) -> str | None:
    """The digest recorded next to an artifact, or ``None`` if absent."""
    side = Path(path).with_suffix(Path(path).suffix + ".sha256")
    if not side.exists():
        return None
    try:
        return side.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def evaluator_identity(root: str | Path | None = None) -> dict[str, Any]:
    """Content identity of the evaluation implementation.

    The identity is the SHA-256 of the per-file digests of the evaluator
    implementation files, so a change to *any* of them (including the
    physics operators the residual evaluators call) changes the identity
    and therefore invalidates the artifacts produced before the change.
    """
    from ..config import project_root

    base = Path(root) if root is not None else project_root()
    files: dict[str, str | None] = {}
    for rel in EVALUATOR_IMPLEMENTATION_FILES:
        p = base / rel
        files[rel] = sha256_file(p) if p.exists() else None
    return {
        "id": "pdelab-evaluation/2",
        "sha256": canonical_json_sha256(files),
        "files": files,
    }


def normalization_identity(stats: Any) -> dict[str, Any]:
    """Content identity of the normalisation constants."""
    payload = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
    return {"values": payload, "sha256": canonical_json_sha256(payload)}


# ----------------------------------------------------------------------
# metric math
# ----------------------------------------------------------------------
def nrmse_per_trajectory(
    prediction: np.ndarray, reference: np.ndarray, epsilon: float = EPSILON
) -> np.ndarray:
    """Per-trajectory relative RMS error.

    ``prediction`` and ``reference`` have shape ``(n_traj, n_snap, ...)``
    and contain **predicted future snapshots only**.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    num = np.sum((prediction - reference) ** 2, axis=tuple(range(1, reference.ndim)))
    den = np.sum(reference**2, axis=tuple(range(1, reference.ndim))) + epsilon
    return np.sqrt(num / den)


def nrmse_per_snapshot(
    prediction: np.ndarray, reference: np.ndarray, epsilon: float = EPSILON
) -> np.ndarray:
    """Per-snapshot relative RMS error, shape (n_traj, n_snap)."""
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    num = np.sum((prediction - reference) ** 2, axis=tuple(range(2, reference.ndim)))
    den = np.sum(reference**2, axis=tuple(range(2, reference.ndim))) + epsilon
    return np.sqrt(num / den)


@dataclass
class ErrorSummary:
    """Aggregate summary of a data-error metric.

    ``n_trajectories`` counts the units the aggregation is over.  For a
    per-trajectory summary that is the number of held-out trajectories;
    for a per-window summary it is the number of windows and
    ``aggregation`` says so explicitly.
    """

    mean: float
    median: float
    std: float
    min: float
    max: float
    n_trajectories: int
    near_zero_reference_count: int
    epsilon: float = EPSILON
    n_windows: int | None = None
    aggregation: str = "per_trajectory"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "n_trajectories": self.n_trajectories,
            "n_windows": self.n_windows,
            "aggregation": self.aggregation,
            "near_zero_reference_count": self.near_zero_reference_count,
            "epsilon": self.epsilon,
        }


def summarize_nrmse(
    per_traj: np.ndarray,
    reference_energy: np.ndarray | None = None,
    epsilon: float = EPSILON,
    aggregation: str = "per_trajectory",
    n_windows: int | None = None,
) -> ErrorSummary:
    per_traj = np.asarray(per_traj, dtype=np.float64)
    near_zero = 0
    if reference_energy is not None:
        near_zero = int(
            np.sum(np.asarray(reference_energy, dtype=np.float64) < NEAR_ZERO_REF_THRESHOLD)
        )
    return ErrorSummary(
        mean=float(np.mean(per_traj)),
        median=float(np.median(per_traj)),
        std=float(np.std(per_traj, ddof=0)),
        min=float(np.min(per_traj)),
        max=float(np.max(per_traj)),
        n_trajectories=int(per_traj.size),
        near_zero_reference_count=near_zero,
        epsilon=epsilon,
        n_windows=n_windows,
        aggregation=aggregation,
    )


def aggregate_windows_per_trajectory(
    per_window_values: np.ndarray, window_trajectory_ids: np.ndarray
) -> np.ndarray:
    """Mean of per-window values within each trajectory.

    ``per_window_values`` holds one value per *window* and
    ``window_trajectory_ids`` the trajectory each window belongs to.  The
    result holds one value per *trajectory*, in ascending trajectory-id
    order.  This is what makes a one-step mean comparable with a
    rollout mean: both are then averages over trajectories.
    """
    values = np.asarray(per_window_values, dtype=np.float64)
    ids = np.asarray(window_trajectory_ids, dtype=np.int64)
    if values.shape[0] != ids.shape[0]:
        raise ValueError("per-window values and trajectory ids must have the same length")
    if ids.size == 0:
        return np.zeros(0, dtype=np.float64)
    n_traj = int(ids.max()) + 1
    totals = np.bincount(ids, weights=values, minlength=n_traj)
    counts = np.bincount(ids, minlength=n_traj).astype(np.float64)
    counts[counts == 0.0] = 1.0
    return totals / counts


# ----------------------------------------------------------------------
# physics residual error
# ----------------------------------------------------------------------
def residual_rms(
    residual: torch.Tensor, weights: torch.Tensor | None = None
) -> float:
    """Physical-unit RMS residual over the declared space/time domain.

    ``residual`` has shape ``(n_samples, n_time, *spatial)`` (the time
    axis is optional). The RMS is taken over the space/time axes for
    every sample, then the root-mean-square is taken over the samples,
    so that every trajectory contributes equally regardless of its
    length and the aggregation is identical for every compared model.
    """
    r = residual.detach().to(torch.float64)
    if r.dim() < 2:
        return float(torch.sqrt(torch.mean(r**2)))
    space_time_axes = tuple(range(1, r.dim()))
    per_sample = torch.sqrt(torch.mean(r**2, dim=space_time_axes))
    return float(torch.sqrt(torch.mean(per_sample**2)))


def relative_residual_improvement(
    baseline_residual: float, proposed_residual: float
) -> float:
    """100 * (baseline - proposed) / baseline, with a guarded denominator.

    A zero or near-zero baseline denominator is reported as ``nan`` and
    must be handled explicitly by the caller rather than silently
    producing a huge percentage.
    """
    if not np.isfinite(baseline_residual) or abs(baseline_residual) < 1e-30:
        return float("nan")
    return float(100.0 * (baseline_residual - proposed_residual) / baseline_residual)


# ----------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------
def persistence_forecast(
    initial: np.ndarray, n_steps: int
) -> np.ndarray:
    """Copy the initial state forward for ``n_steps`` future snapshots.

    ``initial`` has shape ``(n_traj, *spatial)`` and the result has
    shape ``(n_traj, n_steps, *spatial)``. The repeat is performed on
    the *time* axis of the output, which is inserted as axis 1, so the
    spatial axes are never tiled by mistake.
    """
    initial = np.asarray(initial)
    if initial.ndim < 2:
        raise ValueError("initial must have a batch axis and at least one spatial axis")
    return np.repeat(initial[:, None, ...], n_steps, axis=1)


def _burgers_rhs(u: torch.Tensor, nu: float) -> torch.Tensor:
    """Right-hand side of the viscous Burgers equation, ``u_t = F(u)``.

    The spatial discretisation is the same spectral one that the
    residual evaluators use, so the numerical baseline and the residual
    metric see the same operator.
    """
    from ..physics.operators import d2x_1d, dealias_2_3, spectral_k_1d

    n = u.shape[-1]
    k = spectral_k_1d(n, device=u.device, dtype=u.dtype)
    mask = torch.as_tensor(dealias_2_3(n), device=u.device, dtype=torch.bool)
    u_filtered = torch.fft.ifft(torch.fft.fft(u, dim=-1) * mask, dim=-1).real
    flux = 0.5 * u_filtered**2
    flux_x = torch.fft.ifft(1j * k * torch.fft.fft(flux, dim=-1), dim=-1).real
    u_xx = d2x_1d(u, k)
    return -flux_x + nu * u_xx


def _allen_cahn_rhs(u: torch.Tensor, D: float, a: float) -> torch.Tensor:
    """Right-hand side of Allen-Cahn, ``u_t = D laplacian(u) + a (u - u^3)``."""
    from ..physics.operators import laplacian_2d, spectral_k_2d

    kx, ky = spectral_k_2d(
        u.shape[-2], u.shape[-1], device=u.device, dtype=u.dtype
    )
    lap = laplacian_2d(u, kx, ky)
    return D * lap + a * (u - u**3)


def _retained_band_mask(shape: tuple[int, ...], device, dtype) -> torch.Tensor:
    """The 2/3 retained-band mask of the *last* axis, broadcastable."""
    from ..physics.operators import dealias_2_3

    n = shape[-1]
    mask = torch.as_tensor(dealias_2_3(n), device=device, dtype=torch.bool)
    view = [1] * len(shape)
    view[-1] = n
    return mask.reshape(view)


def _galerkin_filter(u: torch.Tensor) -> torch.Tensor:
    """Project a state back onto the retained spectral band.

    A Galerkin truncation (filter the state, not only the product) is
    what makes the explicit spectral stepping stable: the modes above
    the retained band would otherwise be amplified by the diffusion.
    """
    mask = _retained_band_mask(tuple(u.shape), u.device, u.dtype)
    if u.dim() == 2:  # 1D
        return torch.fft.ifft(torch.fft.fft(u, dim=-1) * mask, dim=-1).real
    if u.dim() == 3:  # 2D
        mask2 = mask.reshape(1, 1, -1)
        return torch.fft.ifft2(torch.fft.fft2(u, dim=(-2, -1)) * mask2, dim=(-2, -1)).real
    raise ValueError("unsupported field shape")


def numerical_stepping_plan(
    system: str,
    params: dict[str, float],
    dt: float,
    n_spatial: int,
    u_max: float,
) -> dict[str, Any]:
    """The stability-limited stepping plan of the numerical baseline.

    An explicit spectral scheme is only stable while the step size
    resolves the fastest *retained* mode.  The plan therefore fixes the
    number of internal sub-steps from the diffusion limit, the
    advection limit and the reaction limit, and it records the limits
    so that the cost of the baseline is visible next to its accuracy.
    """
    params = {k: float(v) for k, v in (params or {}).items()}
    n = int(n_spatial)
    if system == "burgers":
        nu = params.get("nu", 0.01)
        k_max = 2.0 * np.pi * max(1, n // 3)
        limits = {}
        if nu > 0:
            limits["diffusion"] = 1.0 / (nu * k_max**2)
        if u_max > 0:
            limits["advection"] = 1.0 / (k_max * u_max)
        reaction = None
    elif system == "allen_cahn":
        D = params.get("D", 0.001)
        a = params.get("a", 1.0)
        k_max = 2.0 * np.pi * max(1, n // 3)
        limits = {}
        if D > 0:
            limits["diffusion"] = 1.0 / (D * k_max**2)
        reaction = 1.0 / a if a > 0 else None
        if reaction is not None:
            limits["reaction"] = reaction
    else:
        raise ValueError(f"unknown system {system!r}")

    dt_max = min(limits.values()) if limits else float(dt)
    dt_max = min(dt_max, float(dt))
    sub_steps = max(1, int(np.ceil(float(dt) / (0.5 * dt_max)))) if dt_max > 0 else 1
    return {
        "system": system,
        "observation_dt": float(dt),
        "internal_dt": float(dt) / sub_steps,
        "sub_steps_per_observation_step": int(sub_steps),
        "stability_limits": {k: float(v) for k, v in limits.items()},
        "max_speed": float(u_max),
        "spatial_filter": "2/3 retained band (Galerkin truncation)",
    }


def numerical_stepping_forecast(
    initial: np.ndarray,
    n_steps: int,
    dt: float,
    system: str,
    params: dict[str, float],
    *,
    scheme: str = "rk2",
    sub_steps: int | None = None,
) -> np.ndarray:
    """Advance the PDE itself on the observation grid (a numerical baseline).

    The state is advanced with an explicit time-stepping scheme on the
    *observation* grid, starting from the initial frame, for the same
    number of steps and with the same ``dt`` as the learned model's
    rollout.  The result therefore lives on exactly the same grid and
    horizon as the model's prediction and can be compared with it
    directly.

    The scheme is stabilised the way a spectral Galerkin method has to
    be: the state is projected back onto the retained band after every
    internal step, and the observation interval is split into
    stability-limited sub-steps (see :func:`numerical_stepping_plan`).

    Interface difference from :func:`persistence_forecast`
    ---------------------------------------------------------
    * ``persistence_forecast(initial, n_steps)`` needs **only** the
      initial frame.  It is parameter free and knows nothing about the
      PDE.
    * ``numerical_stepping_forecast(initial, n_steps, dt, system,
      params)`` additionally needs the observation time step ``dt``, the
      identity of the system and its physical parameters (``nu`` for
      Burgers, ``D`` and ``a`` for Allen-Cahn).  It is therefore a
      *parameter-informed* baseline: it is not a learned model and it is
      not comparable with one in terms of prior knowledge.  It exists to
      quantify how much of the task is solvable by direct numerical
      integration at negligible cost, and its cost is reported next to
      its accuracy.

    Parameters
    ----------
    scheme:
        ``"rk2"`` (explicit trapezoid / Heun, the default) or
        ``"euler"`` (explicit Euler).
    sub_steps:
        Internal sub-steps per observation step.  ``None`` selects the
        stability-limited number from :func:`numerical_stepping_plan`.
    """
    if n_steps < 0:
        raise ValueError("n_steps must not be negative")
    if dt <= 0:
        raise ValueError("dt must be positive")
    initial = np.asarray(initial, dtype=np.float32)
    if initial.ndim < 2:
        raise ValueError("initial must have a batch axis and at least one spatial axis")
    params = {k: float(v) for k, v in (params or {}).items()}

    u = torch.from_numpy(initial).to(torch.float64)
    if sub_steps is None:
        plan = numerical_stepping_plan(
            system, params, dt, int(initial.shape[-1]), float(u.abs().max())
        )
        sub_steps = int(plan["sub_steps_per_observation_step"])
    sub_steps = max(1, int(sub_steps))
    dt_internal = float(dt) / sub_steps

    def rhs(state: torch.Tensor) -> torch.Tensor:
        if system == "burgers":
            return _burgers_rhs(state, params.get("nu", 0.01))
        if system == "allen_cahn":
            return _allen_cahn_rhs(state, params.get("D", 0.001), params.get("a", 1.0))
        raise ValueError(f"unknown system {system!r}")

    out = np.zeros((initial.shape[0], n_steps, *initial.shape[1:]), dtype=np.float32)
    for step in range(n_steps):
        for _ in range(sub_steps):
            if scheme == "euler":
                u = u + dt_internal * rhs(u)
            elif scheme == "rk2":
                k1 = rhs(u)
                k2 = rhs(u + dt_internal * k1)
                u = u + 0.5 * dt_internal * (k1 + k2)
            else:
                raise ValueError(f"unknown scheme {scheme!r}")
            u = _galerkin_filter(u)
        out[:, step] = u.to(torch.float32).numpy()
    return out


def mean_conservation_error(
    trajectory: np.ndarray, reference_mean: float
) -> float:
    """Max absolute deviation of the spatial mean from the initial mean."""
    traj = np.asarray(trajectory, dtype=np.float64)
    axis = tuple(range(1, traj.ndim))
    means = traj.mean(axis=axis)
    return float(np.max(np.abs(means - reference_mean)))
