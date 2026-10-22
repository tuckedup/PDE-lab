"""One-step metric check for the PDELab release.

This fixture runs on **CPU only**, from a directory **outside** the
source checkout, against the **installed wheel**.  It exercises exactly
the code path that produces the one-step nRMSE of the study:

    teacher_forced_one_step -> nrmse_per_trajectory -> summarize_nrmse

and the two model-free baselines that the study always reports next to
it: the persistence forecast and the direct numerical stepping of the
PDE itself.

Nothing is trained and no dataset is read: the fixture builds a tiny
synthetic trajectory set whose one-step error is known analytically, so
the check is deterministic and finishes in a couple of seconds.

Exit code 0 means every assertion held.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

import pdelab
from pdelab.data.dataset import NormalizationStats
from pdelab.evaluation.metrics import (
    aggregate_windows_per_trajectory,
    nrmse_per_trajectory,
    numerical_stepping_forecast,
    persistence_forecast,
    relative_residual_improvement,
    summarize_nrmse,
)
from pdelab.evaluation.rollout import teacher_forced_one_step
from pdelab.models.fno import FNO1d

CPU_THREADS = 4
N_TRAJ = 3
N_SNAP = 5
N_GRID = 16

checks: list[dict] = []


def check(name: str, ok: bool, detail: dict) -> None:
    checks.append({"name": name, "passed": bool(ok), **detail})
    print(f"[one_step_metric_check] {'PASS' if ok else 'FAIL'}  {name}")


class TinyTrajectoryDataset:
    """A minimal stand-in for ``TrajectoryDataset`` (no HDF5 file)."""

    def __init__(self, u: np.ndarray, history: int = 1) -> None:
        self.u = u
        self.history = history
        self.n_traj, self.n_times = u.shape[0], u.shape[1]
        self.n_windows = self.n_times - history
        self._index = [
            (t, s)
            for t in range(self.n_traj)
            for s in range(0, self.n_windows)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj, start = self._index[idx]
        hist = self.u[traj, start : start + self.history]
        target = self.u[traj, start + self.history][np.newaxis, ...]
        return {
            "history": torch.from_numpy(np.ascontiguousarray(hist)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "traj": torch.tensor(traj, dtype=torch.long),
            "time_index": torch.tensor(start, dtype=torch.long),
        }


def main() -> int:
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(20261022)
    np.random.seed(20261022)

    module_file = os.path.abspath(pdelab.__file__)
    print(f"[one_step_metric_check] pdelab.__file__ = {module_file}")
    print(f"[one_step_metric_check] pdelab.__version__ = {pdelab.__version__}")
    print(f"[one_step_metric_check] torch threads = {torch.get_num_threads()}")
    print(f"[one_step_metric_check] sys.path[0] = {sys.path[0]!r}")

    # 1. the import must come from the installed wheel, not from a
    #    source checkout that happens to be on sys.path.
    from_source_tree = os.sep + "src" + os.sep + "pdelab" + os.sep in module_file
    check(
        "pdelab_is_imported_from_the_installed_wheel",
        not from_source_tree,
        {"pdelab_file": module_file, "looks_like_source_tree": from_source_tree},
    )

    # 2. the metric itself, against an analytically known value.
    #    prediction = 1.1 * reference  =>  nRMSE = |1.1 - 1| = 0.1
    rng = np.random.default_rng(7)
    reference = rng.standard_normal((2, 3, 8)) + 3.0
    prediction = 1.1 * reference
    per_traj = nrmse_per_trajectory(prediction, reference)
    check(
        "nrmse_matches_the_analytic_value",
        bool(np.allclose(per_traj, 0.1, rtol=0.0, atol=1e-12)),
        {
            "expected": 0.1,
            "measured": [float(v) for v in per_traj],
            "formula": "sqrt(sum((P-R)^2) / (sum(R^2) + 1e-12))",
        },
    )

    # 3. the one-step path of the real evaluation code.
    u = np.zeros((N_TRAJ, N_SNAP, N_GRID), dtype=np.float32)
    for t in range(N_SNAP):
        u[:, t, :] = np.cos(2.0 * np.pi * np.arange(N_GRID) / N_GRID) * np.exp(
            -0.2 * t
        ) + 0.1 * t
    dataset = TinyTrajectoryDataset(u, history=1)
    stats = NormalizationStats(mean=0.0, std=1.0, source="fixture")
    model = FNO1d(modes=4, width=4, history=1, n_layers=2)
    preds, refs, traj_ids = teacher_forced_one_step(
        model, dataset, stats, params=None, device="cpu"
    )
    window_values = nrmse_per_trajectory(preds, refs)
    energy = np.sum(refs.astype(np.float64) ** 2, axis=(1, 2))
    summary = summarize_nrmse(
        window_values, energy, aggregation="per_window", n_windows=len(dataset)
    )
    per_traj_values = aggregate_windows_per_trajectory(window_values, traj_ids)
    ok_shape = preds.shape == refs.shape == (N_TRAJ * (N_SNAP - 1), 1, N_GRID)
    check(
        "teacher_forced_one_step_produces_the_expected_shapes",
        ok_shape
        and len(window_values) == len(dataset)
        and int(summary.n_windows) == len(dataset),
        {
            "predictions_shape": list(preds.shape),
            "references_shape": list(refs.shape),
            "n_windows": int(summary.n_windows),
            "one_step_nrmse_per_window_mean": float(summary.mean),
            "one_step_nrmse_per_trajectory_mean": float(np.mean(per_traj_values)),
        },
    )
    check(
        "one_step_nrmse_is_finite_and_non_negative",
        bool(np.all(np.isfinite(window_values)) and np.all(window_values >= 0.0)),
        {
            "min": float(np.min(window_values)),
            "max": float(np.max(window_values)),
            "mean": float(summary.mean),
        },
    )

    # 4. the persistence baseline: exact on a constant field.
    const = np.full((2, 4, 8), 0.25, dtype=np.float32)
    persistence = persistence_forecast(const[:, 0, :], n_steps=3)
    persistence_nrmse = nrmse_per_trajectory(persistence, const[:, 1:, :])
    check(
        "persistence_baseline_is_exact_on_a_constant_field",
        bool(np.allclose(persistence_nrmse, 0.0, rtol=0.0, atol=1e-6)),
        {
            "persistence_shape": list(persistence.shape),
            "nrmse": [float(v) for v in persistence_nrmse],
        },
    )

    # 5. the direct numerical stepping baseline (the parameter informed
    #    one that the study compares against): it must advance the PDE,
    #    keep the shape, stay finite and conserve the spatial mean.
    n_step = 32
    x = np.arange(n_step) / n_step
    ic = (0.8 * np.cos(2.0 * np.pi * x) + 0.3 * np.cos(4.0 * np.pi * x + 0.5)).astype(
        np.float32
    )
    stepped = numerical_stepping_forecast(
        ic[None, :],
        n_steps=2,
        dt=0.005,
        system="burgers",
        params={"nu": 0.01},
    )
    mean_drift = float(np.max(np.abs(stepped.mean(axis=-1) - ic.mean())))
    evolved = bool(np.max(np.abs(stepped[:, 1, :] - ic[None, :])) > 0.0)
    check(
        "numerical_stepping_baseline_advances_the_pde_and_keeps_the_mean",
        stepped.shape == (1, 2, n_step)
        and bool(np.all(np.isfinite(stepped)))
        and evolved
        and mean_drift < 1e-5,
        {
            "shape": list(stepped.shape),
            "max_abs_mean_drift": mean_drift,
            "state_evolved": evolved,
            "scheme": "rk2 with 2/3 Galerkin truncation",
        },
    )

    # 6. the residual improvement helper used by the report.
    improvement = relative_residual_improvement(1.0, 0.5)
    check(
        "relative_residual_improvement_is_correct",
        abs(improvement - 50.0) < 1e-12,
        {"baseline": 1.0, "proposed": 0.5, "improvement_percent": improvement},
    )

    payload = {
        "pdelab_file": module_file,
        "pdelab_version": pdelab.__version__,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "cuda_available": bool(torch.cuda.is_available()),
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }
    print("[one_step_metric_check] " + json.dumps(payload, indent=2))
    print(
        "[one_step_metric_check] OVERALL: "
        + ("PASS" if payload["passed"] else "FAIL")
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
