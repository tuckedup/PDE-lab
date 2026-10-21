"""Evaluate a trained run on the held-out test split.

Reports, for the same protocol used for every compared condition:

* free-rollout nRMSE (the primary data metric),
* teacher-forced one-step nRMSE (per trajectory, with the per-window
  summary reported separately and labelled as such),
* the physical-unit PDE residual error of the rolled-out trajectory
  under **every** declared residual evaluator, together with the
  reference solution's own residual under the same evaluator,
* the persistence baseline and a low-cost numerical stepping baseline
  under the identical protocol, with their runtimes,
* per-time error curves and per-trajectory values.

Residual evaluators
-------------------
Two evaluators are declared and never mixed:

``implicit_euler_first_order`` (primary)
    ``u_t = (u_i - u_{i-1}) / dt`` with the spatial operator evaluated
    at ``u_i``.  This is the implicit-Euler form of the semidiscrete
    system: the new state appears on both sides.  It is first order in
    time, so even an exact solution sampled at ``dt`` shows a non-zero
    residual.  That value is *measured*, not asserted: the temporal
    refinement probe below quantifies how the reference solution's own
    residual scales with ``dt``.

``centred_first_derivative`` (secondary diagnostic)
    ``u_t = (u_{i+1} - u_{i-1}) / (2 dt)`` with the spatial operator
    evaluated at ``u_i``.  This is a genuine centred **first**
    derivative, second order in time.

Both evaluators are evaluated on the **same shared support**: the
interior snapshot indices ``1 .. T-2`` of the trajectory.  The centred
stencil additionally touches ``u_{i+1}``; the evaluated *times* are
identical, so the two numbers are directly comparable.

The operators themselves come from ``pdelab.physics.operators`` (owned
by the numerical/data specialist) when the corrected centred
first-derivative residual and the first-order implicit-Euler residual
are available there; otherwise the local reference implementation in
this module is used.  Every artifact records which implementation was
used and cross-checks the two against each other.

Content identity
----------------
Every artifact records the SHA-256 of the checkpoint, of the evaluated
dataset split, of the normalisation constants, of the evaluator
implementation, of the resolved evaluation settings and of the
protocol.  Artifacts are written atomically and the digest of the
finished bytes is stored next to them.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import project_root
from ..data.dataset import (
    NormalizationStats,
    TrajectoryDataset,
    split_paths,
)
from ..evaluation.metrics import (
    EPSILON,
    METRIC_IMPLEMENTATION_VERSION,
    NEAR_ZERO_REF_THRESHOLD,
    aggregate_windows_per_trajectory,
    canonical_json_sha256,
    evaluator_identity,
    normalization_identity,
    nrmse_per_snapshot,
    nrmse_per_trajectory,
    numerical_stepping_forecast,
    numerical_stepping_plan,
    persistence_forecast,
    residual_rms,
    sha256_file,
    summarize_nrmse,
    write_json_atomic,
)
from ..evaluation.rollout import rollout, teacher_forced_one_step
from ..models.registry import build_model_from_config

#: Identifier of the evaluation protocol.  Bumping it invalidates every
#: artifact produced under the previous protocol.
PROTOCOL_ID = "pdelab-free-rollout/2"

#: The two declared residual evaluators, in reporting order.
RESIDUAL_EVALUATORS = ("implicit_euler_first_order", "centred_first_derivative")

#: The shared support of both evaluators.
RESIDUAL_SUPPORT = "interior_snapshots_only"

#: Names of the operators that the numerical/data specialist provides in
#: ``pdelab.physics.operators``.  The first name that exists is used;
#: when none exists the local reference implementation in this module is
#: used instead (and the artifact records that).
#:
#: The implicit-Euler candidate list ends with the operators that already
#: exist: ``burgers_residual`` / ``allen_cahn_residual`` *are* the
#: first-order implicit-Euler residual
#: (``(u_next - u_prev)/dt`` with the spatial operator evaluated at
#: ``u_next``).  They are the fallback until the specialist's
#: explicitly named versions land.
CENTRED_FIRST_OPERATOR_NAMES = {
    "burgers": (
        "burgers_residual_centered_first_derivative",
        "burgers_residual_centred_first_derivative",
        "burgers_residual_centered_first",
        "burgers_residual_centred_first",
    ),
    "allen_cahn": (
        "allen_cahn_residual_centered_first_derivative",
        "allen_cahn_residual_centred_first_derivative",
        "allen_cahn_residual_centered_first",
        "allen_cahn_residual_centred_first",
    ),
}

IMPLICIT_EULER_OPERATOR_NAMES = {
    "burgers": (
        "burgers_residual_implicit_euler",
        "burgers_residual_implicit_euler_first_order",
        "burgers_residual",
    ),
    "allen_cahn": (
        "allen_cahn_residual_implicit_euler",
        "allen_cahn_residual_implicit_euler_first_order",
        "allen_cahn_residual",
    ),
}


# ----------------------------------------------------------------------
# residual evaluators
# ----------------------------------------------------------------------
def _spatial_residual_term(
    system: str, u: torch.Tensor, params: dict[str, float]
) -> torch.Tensor:
    """The spatial part of the PDE residual, evaluated at ``u``.

    ``burgers``:  ``(u^2/2)_x - nu u_xx``
    ``allen_cahn``: ``-D laplacian(u) - a (u - u^3)``

    The discretisation is the one of the physics operators in
    ``pdelab.physics.operators`` (spectral in space, retained-band
    products), so that both residual evaluators see the same spatial
    operator and differ only in their time stencil.  The retained-band
    helpers are used when the physics module provides them, so the local
    reference implementation and the operators of the physics module
    cannot drift apart.
    """
    from ..physics import operators as ops

    if system == "burgers":
        nu = float(params.get("nu", 0.01))
        n = u.shape[-1]
        k = ops.spectral_k_1d(n, device=u.device, dtype=u.dtype)
        if hasattr(ops, "burgers_flux_dealiased"):
            flux = ops.burgers_flux_dealiased(u)
        else:  # pragma: no cover - older physics module
            mask = torch.as_tensor(
                ops.dealias_2_3(n), device=u.device, dtype=torch.bool
            )
            u_filtered = torch.fft.ifft(torch.fft.fft(u, dim=-1) * mask, dim=-1).real
            flux = 0.5 * u_filtered**2
        flux_x = torch.fft.ifft(1j * k * torch.fft.fft(flux, dim=-1), dim=-1).real
        u_xx = ops.d2x_1d(u, k)
        return flux_x - nu * u_xx
    if system == "allen_cahn":
        D = float(params.get("D", 0.001))
        a = float(params.get("a", 1.0))
        kx, ky = ops.spectral_k_2d(
            u.shape[-2], u.shape[-1], device=u.device, dtype=u.dtype
        )
        lap = ops.laplacian_2d(u, kx, ky)
        if hasattr(ops, "allen_cahn_cubic_dealiased"):
            cubic = ops.allen_cahn_cubic_dealiased(u)
        else:  # pragma: no cover - older physics module
            cubic = u**3
        return -D * lap - a * (u - cubic)
    raise ValueError(f"unknown system {system!r}")


def _local_residual_tensor(
    system: str,
    traj: torch.Tensor,
    dt: float,
    params: dict[str, float],
    evaluator: str,
) -> torch.Tensor:
    """Residual of a trajectory on the shared interior support.

    ``traj`` has shape ``(B, T, *spatial)`` in physical units.  The
    result has shape ``(B, T-2, *spatial)``: one residual value per
    interior snapshot index.
    """
    if traj.dim() < 3:
        raise ValueError("a trajectory needs a batch axis, a time axis and a spatial axis")
    if traj.shape[1] < 3:
        raise ValueError(
            "the interior support needs at least three snapshots, "
            f"got {traj.shape[1]}"
        )
    centre = traj[:, 1:-1]
    if evaluator == "centred_first_derivative":
        u_t = (traj[:, 2:] - traj[:, :-2]) / (2.0 * dt)
    elif evaluator == "implicit_euler_first_order":
        u_t = (traj[:, 1:-1] - traj[:, :-2]) / dt
    else:
        raise ValueError(f"unknown residual evaluator {evaluator!r}")
    return u_t + _spatial_residual_term(system, centre, params)


def _resolve_operator(candidates: tuple[str, ...]):
    """The first candidate that exists in ``pdelab.physics.operators``."""
    from ..physics import operators as ops

    for name in candidates:
        fn = getattr(ops, name, None)
        if callable(fn):
            return fn, name
    return None, None


def _align_support(res: torch.Tensor, n_times: int) -> torch.Tensor:
    """Cut an operator's output down to the shared interior support."""
    if res.dim() < 2:
        raise ValueError("the operator returned a tensor without a time axis")
    n = int(res.shape[1])
    if n == n_times - 2:
        return res
    if n == n_times:
        return res[:, 1:-1]
    if n == n_times - 1:
        return res[:, :-1]
    raise ValueError(
        f"the operator returned {n} time slices; expected {n_times - 2} "
        "(the shared interior support)"
    )


def _call_operator(fn, system: str, traj: torch.Tensor, dt: float, params: dict[str, float]):
    """Call a physics operator, adapting to the signature it declares.

    The operator may take the whole trajectory (and return the interior
    residuals) or a pair of consecutive snapshots.  The signature is
    inspected so that neither convention is guessed.
    """
    sig = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    if "dt" in sig.parameters:
        kwargs["dt"] = float(dt)
    if system == "burgers" and "nu" in sig.parameters:
        kwargs["nu"] = float(params.get("nu", 0.01))
    if system == "allen_cahn":
        if "D" in sig.parameters:
            kwargs["D"] = float(params.get("D", 0.001))
        if "a" in sig.parameters:
            kwargs["a"] = float(params.get("a", 1.0))
    required = [
        name
        for name, p in sig.parameters.items()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
        and name not in kwargs
    ]
    if len(required) >= 2:
        out = fn(traj[:, 1:-1], traj[:, :-2], **kwargs)
    else:
        out = fn(traj, **kwargs)
    return _align_support(out, int(traj.shape[1]))


def _residual_tensor(
    system: str,
    trajectory: np.ndarray,
    dt: float,
    params: dict[str, float],
    evaluator: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Residual of a trajectory under one declared evaluator.

    Returns the residual tensor on the shared interior support and a
    provenance record naming the implementation that produced it and the
    result of the cross-check against the local reference
    implementation.
    """
    traj = torch.as_tensor(np.asarray(trajectory))
    if not torch.is_floating_point(traj):
        traj = traj.to(torch.float32)
    local = _local_residual_tensor(system, traj, dt, params, evaluator)
    local_rms = residual_rms(local)

    names = (
        IMPLICIT_EULER_OPERATOR_NAMES
        if evaluator == "implicit_euler_first_order"
        else CENTRED_FIRST_OPERATOR_NAMES
    )
    fn, name = _resolve_operator(names[system])
    info: dict[str, Any] = {
        "evaluator": evaluator,
        "support": RESIDUAL_SUPPORT,
        "support_indices": "1 .. T-2 (interior snapshots of the trajectory)",
        "shared_with": list(RESIDUAL_EVALUATORS),
    }
    if fn is None:
        info["implementation"] = "evaluation.evaluate.local_reference"
        info["operator"] = None
        info["cross_check"] = {"available": False}
        return local, info

    try:
        out = _call_operator(fn, system, traj, dt, params)
    except Exception as exc:  # the operator's convention is unknown
        info["implementation"] = "evaluation.evaluate.local_reference"
        info["operator"] = name
        info["operator_error"] = f"{type(exc).__name__}: {exc}"
        info["cross_check"] = {"available": False}
        return local, info

    rms = residual_rms(out)
    denom = max(abs(local_rms), 1e-30)
    rel = abs(rms - local_rms) / denom
    info["implementation"] = f"physics.operators.{name}"
    info["operator"] = name
    info["cross_check"] = {
        "available": True,
        "local_reference_rms": local_rms,
        "operator_rms": rms,
        "relative_difference": rel,
        "agrees": bool(rel <= 1e-3),
        "tolerance": 1e-3,
    }
    return out, info


def _residual_rms(
    system: str,
    trajectory: np.ndarray,
    dt: float,
    params: dict[str, float],
    evaluator: str,
) -> tuple[float, dict[str, Any]]:
    """RMS residual of a trajectory under one declared evaluator."""
    tensor, info = _residual_tensor(system, trajectory, dt, params, evaluator)
    return residual_rms(tensor), info


def _probe_solution(
    system: str, params: dict[str, float], times: np.ndarray, n_spatial: int
) -> tuple[np.ndarray, np.ndarray]:
    """An exact solution of the system PDE and its exact residual field.

    ``allen_cahn``
        The spatially uniform reaction solution
        ``u(t) = 1/sqrt(1 + (u0^-2 - 1) exp(-2 a t))``, which solves
        ``u_t = a (u - u^3)`` exactly.  The Laplacian of a uniform field
        vanishes, so this is an exact solution of the full Allen-Cahn
        equation on the observation grid.

    ``burgers``
        A manufactured single-mode solution
        ``u(x, t) = A exp(-lambda t) sin(2 pi m x)`` together with the
        forcing ``F = u_t + (u^2/2)_x - nu u_xx`` evaluated with the
        same spectral operators.  Subtracting the forcing leaves the
        *temporal* truncation error alone.

    Returns ``(trajectory, exact_residual)`` where ``trajectory`` has
    shape ``(1, n_times, *spatial)`` and ``exact_residual`` the residual
    that the *exact* solution has at the interior snapshot indices
    (zero for a solution of the PDE, the forcing for a manufactured
    one).  Subtracting it from the evaluated residual leaves the
    temporal truncation error alone.
    """
    if system == "allen_cahn":
        a = float(params.get("a", 1.0))
        u0 = 0.2
        u = 1.0 / np.sqrt(1.0 + (1.0 / u0**2 - 1.0) * np.exp(-2.0 * a * times))
        traj = np.broadcast_to(
            u[None, :, None, None], (1, len(times), n_spatial, n_spatial)
        ).astype(np.float64)
        # The uniform field solves the full Allen-Cahn equation exactly,
        # so its exact residual is zero and the evaluated residual *is*
        # the truncation error of the time stencil.
        exact = np.zeros((1, len(times) - 2, n_spatial, n_spatial), dtype=np.float64)
        return traj, exact

    if system == "burgers":
        A, lam, m = 0.5, 1.0, 1
        x = np.arange(n_spatial, dtype=np.float64) / n_spatial
        amp = A * np.exp(-lam * times)
        u = amp[:, None] * np.sin(2.0 * np.pi * m * x[None, :])
        traj = u[None, ...].astype(np.float64)
        centre = torch.from_numpy(traj[:, 1:-1])
        exact = (-lam * centre) + _spatial_residual_term(
            "burgers", centre, {k: float(v) for k, v in params.items()}
        )
        return traj, exact.numpy()

    raise ValueError(f"unknown system {system!r}")


def residual_temporal_refinement_probe(
    system: str,
    params: dict[str, float],
    dt_ladder: list[float],
    horizon: float | None = None,
    n_spatial: int = 32,
) -> dict[str, Any]:
    """Quantify the temporal truncation error of every residual evaluator.

    For every ``dt`` in the ladder the *exact* solution of the system is
    sampled over the **same** physical time window, the residual is
    evaluated with the declared evaluators on the shared interior
    support, and the exact residual is subtracted.  What is left is the
    error the evaluator itself introduces, i.e. the quantity that an
    exact solution would still show.  This replaces an *asserted* "floor"
    with a *measured* one.

    The probe is evaluated in double precision: the centred stencil is a
    difference of nearly equal samples, and in single precision the
    cancellation error would swamp the truncation error being measured.
    """
    ladder = [float(d) for d in dt_ladder]
    if not ladder:
        raise ValueError("the dt ladder must not be empty")
    window = float(horizon) if horizon is not None else 20.0 * min(ladder)
    rows: list[dict[str, Any]] = []
    for dt in ladder:
        n_snapshots = int(round(window / dt)) + 1
        times = np.arange(n_snapshots, dtype=np.float64) * dt
        grid = n_spatial if system == "burgers" else max(8, n_spatial // 2)
        traj, exact = _probe_solution(system, params, times, grid)
        exact_t = torch.from_numpy(np.asarray(exact, dtype=np.float64))
        entry: dict[str, Any] = {
            "dt": float(dt),
            "n_snapshots": n_snapshots,
            "n_interior_snapshots": n_snapshots - 2,
        }
        for evaluator in RESIDUAL_EVALUATORS:
            tensor, _info = _residual_tensor(
                system, traj, float(dt), params, evaluator
            )
            entry[f"{evaluator}_residual_rms"] = residual_rms(tensor)
            entry[f"{evaluator}_truncation_rms"] = residual_rms(tensor - exact_t)
        rows.append(entry)

    for i in range(1, len(rows)):
        for evaluator in RESIDUAL_EVALUATORS:
            coarse = rows[i - 1][f"{evaluator}_truncation_rms"]
            fine = rows[i][f"{evaluator}_truncation_rms"]
            order = (
                float(np.log2(coarse / fine))
                if fine > 0 and coarse > 0
                else float("nan")
            )
            rows[i][f"observed_order_{evaluator}"] = order

    def _final_order(evaluator: str) -> float | None:
        orders = [
            r[f"observed_order_{evaluator}"]
            for r in rows
            if f"observed_order_{evaluator}" in r
        ]
        return orders[-1] if orders else None

    return {
        "description": (
            "Temporal truncation error of each residual evaluator, measured "
            "on an exact solution of the system PDE. The reported value is "
            "the residual an exact solution still shows under the "
            "evaluator, i.e. the empirical floor of the metric."
        ),
        "support": RESIDUAL_SUPPORT,
        "rows": rows,
        "observed_order_implicit_euler_first_order": _final_order(
            "implicit_euler_first_order"
        ),
        "observed_order_centred_first_derivative": _final_order(
            "centred_first_derivative"
        ),
    }


def residual_rms_for_trajectory(
    system: str,
    trajectory: np.ndarray,
    dt: float,
    params: dict[str, float],
    evaluator: str,
) -> tuple[float, dict[str, Any]]:
    """RMS residual of a trajectory under one declared evaluator.

    ``trajectory`` has shape ``(B, T, *spatial)`` in physical units.  The
    returned value is the RMS over the shared interior support (the
    snapshot indices ``1 .. T-2``).
    """
    return _residual_rms(
        system,
        trajectory,
        float(dt),
        {k: float(v) for k, v in (params or {}).items()},
        evaluator,
    )


def residual_tensor_for_trajectory(
    system: str,
    trajectory: np.ndarray,
    dt: float,
    params: dict[str, float],
    evaluator: str,
) -> torch.Tensor:
    """The residual tensor of a trajectory under one declared evaluator.

    Shape ``(B, T-2, *spatial)``: one residual value per interior
    snapshot index.  Both declared evaluators return the same support,
    so their values are directly comparable.
    """
    traj = torch.as_tensor(np.asarray(trajectory))
    if not torch.is_floating_point(traj):
        traj = traj.to(torch.float32)
    return _local_residual_tensor(
        system,
        traj,
        float(dt),
        {k: float(v) for k, v in (params or {}).items()},
        evaluator,
    )


# ----------------------------------------------------------------------
# the evaluation itself
# ----------------------------------------------------------------------
def _resolved_settings(
    split: str,
    dt: float,
    n_steps: int,
    n_traj: int,
    history_len: int,
) -> dict[str, Any]:
    """The resolved evaluation settings that a metric is bound to."""
    return {
        "metric_implementation_version": METRIC_IMPLEMENTATION_VERSION,
        "split": split,
        "observation_dt": float(dt),
        "n_rollout_steps": int(n_steps),
        "n_test_trajectories": int(n_traj),
        "history_length": int(history_len),
        "epsilon": EPSILON,
        "near_zero_reference_threshold": NEAR_ZERO_REF_THRESHOLD,
        "copied_initial_frame_included_in_error": False,
        "residual_evaluators": list(RESIDUAL_EVALUATORS),
        "residual_support": RESIDUAL_SUPPORT,
        "baselines": ["persistence", "numerical_stepping_rk2"],
        "numerical_stepping_scheme": "rk2",
        "one_step_aggregation": "per_trajectory",
    }


def evaluate_run(
    run_dir: str | Path,
    device: str = "auto",
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate a trained run on the held-out test split.

    Parameters
    ----------
    split:
        ``"test"`` (default) for the reported held-out metric, or
        ``"val"`` for the validation metric. The validation split is
        used for model selection (for example the physics-penalty
        weight) so that the test split is never used to make a choice.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = project_root() / run_dir
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {"status": "failed", "reason": f"no manifest at {manifest_path}"}
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except json.JSONDecodeError as exc:
        return {"status": "failed", "reason": f"manifest is not valid JSON: {exc}"}

    if manifest.get("status") != "completed":
        return {
            "status": "not_run",
            "reason": f"run status is {manifest.get('status')!r}",
            "run_id": manifest.get("run_id"),
        }

    cfg = manifest["experiment"]
    system = cfg["system"]
    dataset_dir = cfg["dataset_dir"]
    paths = split_paths(dataset_dir, system)
    if split not in paths:
        return {"status": "failed", "reason": f"unknown split {split!r}"}
    split_path = paths[split]
    if not split_path.exists():
        return {"status": "failed", "reason": f"no dataset split at {split_path}"}

    dev = torch.device(
        "cuda" if (device == "auto" and torch.cuda.is_available()) else
        ("cpu" if device == "auto" else device)
    )

    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        return {"status": "failed", "reason": f"no checkpoint at {ckpt_path}"}
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    model = build_model_from_config(
        {
            "architecture": cfg["architecture"],
            "history": cfg.get("history", 1),
            **{k: v for k, v in (cfg.get("model_kwargs") or {}).items()},
        },
        dimension=2 if system == "allen_cahn" else 1,
    )
    model.load_state_dict(ckpt["model"])
    model.to(dev)
    model.eval()

    stats = NormalizationStats.from_dict(ckpt["normalization"])
    history_len = int(cfg.get("history", 1))
    eval_set = TrajectoryDataset(split_path, stats, history=history_len, stride=1)

    obs_times = manifest["dataset_manifest"]["system"]["observation_times"]
    dt = float(obs_times[1] - obs_times[0])
    params = manifest["dataset_manifest"]["system"]["params"]

    started = time.time()
    n_steps = len(obs_times) - history_len

    # ---- free rollout (primary protocol) ----
    spatial = eval_set.u.shape[2:]
    preds = np.zeros((eval_set.n_traj, n_steps, *spatial), dtype=np.float32)
    hist_arr = np.zeros((eval_set.n_traj, history_len, *spatial), dtype=np.float32)
    for t in range(eval_set.n_traj):
        h = torch.from_numpy(eval_set.u[t, :history_len]).unsqueeze(0)
        out = rollout(model, h, n_steps, stats, params=None, device=dev)
        # The model emits (B, 1, *spatial); drop the singleton channel.
        preds[t] = out[0, :, 0].cpu().numpy()
        hist_arr[t] = eval_set.u[t, :history_len]

    reference = eval_set.u[:, history_len : history_len + n_steps].astype(np.float32)

    per_traj = nrmse_per_trajectory(preds, reference)
    energy = np.sum(reference**2, axis=tuple(range(1, reference.ndim)))
    summary = summarize_nrmse(per_traj, energy)
    per_snap = nrmse_per_snapshot(preds, reference)

    # ---- persistence baseline under the identical protocol ----
    # The forecast repeats the *initial* frame along the time axis. The
    # helper is used rather than an inline np.repeat so that the axis
    # being tiled is always the time axis.
    t0 = time.perf_counter()
    persistence = persistence_forecast(hist_arr[:, -1], n_steps).astype(np.float32)
    persistence_walltime = time.perf_counter() - t0
    persistence_per_traj = nrmse_per_trajectory(persistence, reference)
    persistence_summary = summarize_nrmse(persistence_per_traj, energy)

    # ---- low-cost numerical stepping baseline, same grid and horizon ----
    # Interface difference: unlike the persistence baseline, this one is
    # parameter informed - it needs dt, the system and its physical
    # parameters - so it is not comparable with a learned model in terms
    # of prior knowledge.  It is reported to show how much of the task
    # direct numerical integration solves on the observation grid.
    step_plan = numerical_stepping_plan(
        system,
        {k: float(v) for k, v in params.items()},
        dt,
        int(spatial[-1]),
        float(np.max(np.abs(hist_arr[:, -1]))),
    )
    t0 = time.perf_counter()
    numerical = numerical_stepping_forecast(
        hist_arr[:, -1],
        n_steps,
        dt,
        system,
        params,
        scheme="rk2",
        sub_steps=int(step_plan["sub_steps_per_observation_step"]),
    )
    numerical_walltime = time.perf_counter() - t0
    numerical_per_traj = nrmse_per_trajectory(numerical, reference)
    numerical_summary = summarize_nrmse(numerical_per_traj, energy)

    # ---- teacher-forced one-step ----
    tf_pred, tf_ref, tf_traj = teacher_forced_one_step(model, eval_set, stats, device=dev)
    tf_window = nrmse_per_trajectory(tf_pred, tf_ref)
    tf_per_traj = aggregate_windows_per_trajectory(tf_window, tf_traj)
    tf_energy = aggregate_windows_per_trajectory(
        np.sum(tf_ref**2, axis=tuple(range(1, tf_ref.ndim))), tf_traj
    )
    tf_summary = summarize_nrmse(
        tf_per_traj,
        tf_energy,
        aggregation="per_trajectory",
        n_windows=int(tf_window.size),
    )
    tf_window_summary = summarize_nrmse(
        tf_window,
        aggregation="per_window",
        n_windows=int(tf_window.size),
    )

    # ---- physics residual error, under every declared evaluator ----
    residual_values: dict[str, float] = {}
    reference_values: dict[str, float] = {}
    residual_info: dict[str, Any] = {}
    for evaluator in RESIDUAL_EVALUATORS:
        value, info = _residual_rms(system, preds, dt, params, evaluator)
        ref_value, ref_info = _residual_rms(system, reference, dt, params, evaluator)
        residual_values[evaluator] = value
        reference_values[evaluator] = ref_value
        residual_info[evaluator] = info
        residual_info[evaluator]["reference_implementation"] = ref_info.get("implementation")
        residual_info[evaluator]["reference_cross_check"] = ref_info.get("cross_check")

    # ---- the empirical floor of the primary evaluator ----
    probe = residual_temporal_refinement_probe(
        system,
        {k: float(v) for k, v in params.items()},
        [dt, dt / 2.0, dt / 4.0],
        horizon=float(obs_times[-1] - obs_times[0]),
        n_spatial=32 if system == "burgers" else 16,
    )

    # ---- content identity ----
    resolved = _resolved_settings(split, dt, n_steps, int(eval_set.n_traj), history_len)
    identity = {
        "evaluator": evaluator_identity(),
        "protocol": {
            "id": PROTOCOL_ID,
            "sha256": canonical_json_sha256(
                {"id": PROTOCOL_ID, "resolved_settings_sha256": canonical_json_sha256(resolved)}
            ),
        },
        "resolved_settings": resolved,
        "resolved_settings_sha256": canonical_json_sha256(resolved),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "dataset": {
            "split": split,
            "path": str(split_path),
            "sha256": sha256_file(split_path),
            "manifest_sha256": manifest["dataset_manifest"]["files"][split]["sha256"],
        },
        "normalization": normalization_identity(stats),
    }
    dataset_manifest_hash = manifest["dataset_manifest"]["files"][split]["sha256"]
    if dataset_manifest_hash != identity["dataset"]["sha256"]:
        identity["dataset"]["manifest_mismatch"] = True

    protocol = {
        "id": PROTOCOL_ID,
        "type": "free_rollout",
        "split": split,
        "observation_dt": dt,
        "n_rollout_steps": n_steps,
        "n_test_trajectories": int(eval_set.n_traj),
        "history_length": history_len,
        "copied_initial_frame_included_in_error": False,
        "epsilon": EPSILON,
        "residual_evaluators": list(RESIDUAL_EVALUATORS),
        "residual_support": RESIDUAL_SUPPORT,
        "residual_support_detail": (
            "both evaluators are evaluated at the interior snapshot indices "
            "1 .. T-2 of the trajectory, so the compared values share the "
            "same support"
        ),
        "baselines": ["persistence", "numerical_stepping_rk2"],
    }

    result = {
        "status": "completed",
        "evidence_status": "valid",
        "run_id": manifest["run_id"],
        "split": split,
        "system": system,
        "architecture": cfg["architecture"],
        "condition": cfg["condition"],
        "seed": cfg.get("seed"),
        "lambda_pde": cfg.get("lambda_pde", 0.0),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "test_dataset_sha256": dataset_manifest_hash,
        "identity": identity,
        "protocol": protocol,
        "residual_evaluators": residual_info,
        "residual_rms_implicit_euler_first_order": residual_values[
            "implicit_euler_first_order"
        ],
        "reference_residual_rms_implicit_euler_first_order": reference_values[
            "implicit_euler_first_order"
        ],
        "residual_rms_centred_first_derivative": residual_values[
            "centred_first_derivative"
        ],
        "reference_residual_rms_centred_first_derivative": reference_values[
            "centred_first_derivative"
        ],
        # Deprecated aliases of the primary (implicit Euler, first order)
        # residual.  They exist only because scripts/linux/select_lambda.py
        # reads them; the values above are canonical.
        "residual_error_physical_rms": residual_values["implicit_euler_first_order"],
        "reference_residual_physical_rms": reference_values[
            "implicit_euler_first_order"
        ],
        "residual_temporal_refinement_probe": probe,
        "nrmse": summary.to_dict(),
        "nrmse_one_step_teacher_forced": tf_summary.to_dict(),
        "nrmse_one_step_teacher_forced_per_window": tf_window_summary.to_dict(),
        "persistence_baseline_nrmse": persistence_summary.to_dict(),
        "persistence_baseline_walltime_seconds": round(persistence_walltime, 6),
        "numerical_stepping_baseline_nrmse": numerical_summary.to_dict(),
        "numerical_stepping_baseline_walltime_seconds": round(numerical_walltime, 6),
        "numerical_stepping_plan": step_plan,
        "per_time_nrmse": per_snap.mean(axis=0).tolist(),
        "per_trajectory_nrmse": per_traj.tolist(),
        "evaluation_walltime_seconds": round(time.time() - started, 3),
    }
    out_path = run_dir / (f"evaluation_{split}.json" if split != "test" else "evaluation.json")
    digest = write_json_atomic(out_path, result)
    result["artifact_sha256"] = digest
    result["evaluation_path"] = str(out_path)
    return result
