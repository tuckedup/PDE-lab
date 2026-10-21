"""Behavioural probes for the defects this work order repairs.

Every probe states one expectation and checks it against *whatever
implementation is importable*.  That makes the script runnable against
the old source and against the new one, so the before/after evidence is
produced by the same measurement.

    python campaigns/evidence-cohort/old_behaviour_probes.py

The script exits non-zero when a probe fails, so it can be used as a
regression gate.  On the old source every probe fails.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import torch

torch.set_num_threads(4)

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _exact_uniform_allen_cahn(times: np.ndarray, a: float = 1.0, u0: float = 0.2):
    return 1.0 / np.sqrt(1.0 + (1.0 / u0**2 - 1.0) * np.exp(-2.0 * a * times))


def _uniform_trajectory(dt: float, n: int = 4) -> np.ndarray:
    times = np.arange(1, int(round(0.2 / dt)) + 1) * dt
    return np.broadcast_to(
        _exact_uniform_allen_cahn(times)[None, :, None, None], (1, len(times), n, n)
    ).copy()


# ----------------------------------------------------------------------
# B. the residual diagnostic
# ----------------------------------------------------------------------
def probe_b_centred_first_derivative_converges() -> None:
    ev = importlib.import_module("pdelab.evaluation.evaluate")
    values = []
    for dt in (0.02, 0.01, 0.005):
        traj = _uniform_trajectory(dt)
        if hasattr(ev, "residual_rms_for_trajectory"):
            value, _info = ev.residual_rms_for_trajectory(
                "allen_cahn", traj, dt, {"D": 0.001, "a": 1.0},
                "centred_first_derivative",
            )
        else:  # the old implementation: the withdrawn "order2" diagnostic
            initial = np.full((1, 4, 4), 0.2)
            value = ev._pair_residuals(
                "allen_cahn", traj, initial, dt, {"D": 0.001, "a": 1.0}, order=2
            )
        values.append(float(value))
    ratio = values[0] / values[1]
    record(
        "B. the diagnostic converges at O(dt^2) on an exact solution",
        bool(ratio > 2.5),
        f"values = {[f'{v:.6g}' for v in values]}, "
        f"value(0.02)/value(0.01) = {ratio:.3f} (O(dt^2) requires ~4)",
    )


def probe_b_primary_floor_is_measured() -> None:
    ev = importlib.import_module("pdelab.evaluation.evaluate")
    if not hasattr(ev, "residual_temporal_refinement_probe"):
        record(
            "B. the primary evaluator's truncation error is measured",
            False,
            "no temporal refinement probe exists",
        )
        return
    probe = ev.residual_temporal_refinement_probe(
        "allen_cahn", {"D": 0.001, "a": 1.0}, [0.1, 0.05, 0.025],
        horizon=1.0, n_spatial=16,
    )
    order = probe["observed_order_implicit_euler_first_order"]
    floor = probe["rows"][0]["implicit_euler_first_order_truncation_rms"]
    record(
        "B. the primary evaluator's truncation error is measured",
        bool(abs(order - 1.0) < 0.25),
        f"observed order = {order:.3f} (first order), floor at dt=0.1 = {floor:.4g}",
    )


def probe_b_no_order2_name_survives() -> None:
    importlib.import_module("pdelab.evaluation.evaluate")
    hits = []
    for path in Path("src/pdelab").rglob("*.py"):
        text = path.read_text()
        if "order2" in text or "order_2" in text:
            hits.append(str(path))
    record(
        "B. the withdrawn 'order2' identity does not survive in the source",
        not hits,
        f"files still mentioning it: {hits or 'none'}",
    )


# ----------------------------------------------------------------------
# fixtures for the report probes
# ----------------------------------------------------------------------
def _sha256(path: Path) -> str:
    """A local digest helper (the old source has none in the metrics module)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fixture_run(root: Path, run_id: str, nrmse: float, *, system="burgers"):
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    h5 = data_dir / f"{system}_test.h5"
    if not h5.exists():
        rng = np.random.default_rng(3)
        with h5py.File(h5, "w") as fh:
            u = rng.standard_normal((4, 8, 32)).astype(np.float32)
            fh.create_dataset("u", data=u)
            fh.create_dataset("ic_seeds", data=np.arange(4))
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    sha256_file = _sha256

    torch.save({"model": {}, "normalization": {"mean": 0.0, "std": 1.0}},
               run_dir / "best.pt")
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "experiment": {
            "system": system,
            "architecture": "fno1d",
            "condition": "data_only",
            "lambda_pde": 0.0,
            "seed": 0,
            "history": 1,
            "dataset_dir": str(data_dir),
        },
        "dataset_manifest": {
            "system": {"observation_times": [i * 0.01 for i in range(8)],
                       "params": {"nu": 0.01}},
            "files": {"test": {"sha256": sha256_file(h5)}},
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    artifact = {
        "status": "completed",
        "run_id": run_id,
        "split": "test",
        "system": system,
        "architecture": "fno1d",
        "condition": "data_only",
        "seed": 0,
        "checkpoint": str(run_dir / "best.pt"),
        "test_dataset_sha256": sha256_file(h5),
        "protocol": {"type": "free_rollout", "split": "test"},
        "nrmse": {"mean": nrmse, "median": nrmse, "std": 0.0, "min": nrmse,
                  "max": nrmse, "n_trajectories": 4},
        "residual_error_physical_rms": 1.0,
        "reference_residual_physical_rms": 1.2,
    }
    if "evaluator_identity" in dir(importlib.import_module("pdelab.evaluation.metrics")):
        from pdelab.evaluation.metrics import (
            canonical_json_sha256,
            evaluator_identity,
        )
        artifact["identity"] = {
            "evaluator": evaluator_identity(),
            "protocol": {"id": "pdelab-free-rollout/2", "sha256": "proto"},
            "resolved_settings": {"split": "test"},
            "resolved_settings_sha256": canonical_json_sha256({"split": "test"}),
            "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
            "dataset": {"split": "test", "path": str(h5), "sha256": sha256_file(h5)},
            "normalization": {"mean": 0.0, "std": 1.0},
        }
        artifact["residual_rms_implicit_euler_first_order"] = 1.0
        artifact["reference_residual_rms_implicit_euler_first_order"] = 1.2
    (run_dir / "evaluation.json").write_text(json.dumps(artifact, indent=2))
    return run_dir, h5


TARGETS = {
    "targets": [
        {"name": "burgers_nrmse", "system": "burgers", "metric": "nrmse",
         "threshold": 0.05, "comparison": "<", "units": "dimensionless"}
    ]
}


def _cohort(run_ids, dataset_sha256):
    metrics = importlib.import_module("pdelab.evaluation.metrics")
    evaluator = {"id": "pdelab-evaluation/2", "sha256": "probe"}
    if hasattr(metrics, "evaluator_identity"):
        evaluator = metrics.evaluator_identity()
    return {
        "cohort_id": "probe-cohort",
        "evaluator": evaluator,
        "protocol": {"id": "pdelab-free-rollout/2", "split": "test"},
        "datasets": {"burgers": {"test_sha256": dataset_sha256}},
        "members": [
            {
                "id": r,
                "system": "burgers",
                "architecture": "fno1d",
                "condition": "data_only",
                "lambda_pde": 0.0,
                "seeds": [0],
                "run_ids": [r],
            }
            for r in run_ids
        ],
        "targets": [
            {"name": "burgers_nrmse", "metric": "nrmse", "members": list(run_ids)}
        ],
    }


def _target_table(report, runs, targets, cohort):
    try:
        return report.build_target_table(runs, targets, cohort)
    except TypeError:
        return report.build_target_table(runs, targets)


# ----------------------------------------------------------------------
# G. the report must not select the best test run
# ----------------------------------------------------------------------
def probe_g_report_does_not_select_the_best_run() -> None:
    report = importlib.import_module("pdelab.reporting.report")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fixture_run(root, "run_good", 0.01)
        _fixture_run(root, "run_bad", 0.20)
        runs = report.collect_runs(root / "runs")
        cohort = _cohort(["run_good", "run_bad"], None)
        rows = _target_table(report, runs, TARGETS, cohort)
        row = rows[0]
        record(
            "G. the report does not pick the best run of the cohort",
            row["status"] != "met",
            f"status = {row['status']!r}, measured = {row['measured']!r} "
            "(the cohort contains a run with nRMSE 0.20, above the 0.05 threshold)",
        )


def probe_g_incomplete_cohort_is_never_met() -> None:
    report = importlib.import_module("pdelab.reporting.report")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _fixture_run(root, "run_only", 0.01)
        runs = report.collect_runs(root / "runs")
        cohort = _cohort(["run_only", "run_missing"], None)
        rows = _target_table(report, runs, TARGETS, cohort)
        row = rows[0]
        record(
            "G. an incomplete cohort is never 'met'",
            row["status"] != "met" and row["measured"] is None,
            f"status = {row['status']!r}, measured = {row['measured']!r}",
        )


def probe_provenance_stale_artifact_is_detected() -> None:
    report = importlib.import_module("pdelab.reporting.report")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir, _h5 = _fixture_run(root, "run_stale", 0.01)
        later = time.time() + 60
        os.utime(run_dir / "best.pt", (later, later))
        runs = report.collect_runs(root / "runs")
        cohort = _cohort(["run_stale"], None)
        rows = _target_table(report, runs, TARGETS, cohort)
        row = rows[0]
        record(
            "provenance. an artifact older than its checkpoint is 'stale'",
            row["status"] == "stale" and row["measured"] is None,
            f"status = {row['status']!r}, measured = {row['measured']!r}",
        )


def probe_provenance_evidence_states_exist() -> None:
    report = importlib.import_module("pdelab.reporting.report")
    states = getattr(report, "EVIDENCE_STATES", ())
    wanted = {"valid", "stale", "invalid_protocol", "failed", "interrupted",
              "not_run", "not_comparable"}
    record(
        "provenance. the evidence states are declared",
        wanted.issubset(set(states)),
        f"EVIDENCE_STATES = {sorted(states) or 'absent'}",
    )


# ----------------------------------------------------------------------
# training: resume, the epoch metadata and the physics gate
# ----------------------------------------------------------------------
def _tiny_dataset(root: Path):
    h5 = root / "train.h5"
    rng = np.random.default_rng(5)
    with h5py.File(h5, "w") as fh:
        u = rng.standard_normal((4, 6, 16)).astype(np.float32)
        u[:, 1:] = 0.95 * u[:, :-1] + 0.05 * rng.standard_normal((4, 5, 16))
        fh.create_dataset("u", data=u)
        fh.create_dataset("ic_seeds", data=np.arange(4))
    return h5


def probe_resume_records_the_current_epoch() -> None:
    from pdelab.data.dataset import NormalizationStats, TrajectoryDataset
    from pdelab.models.fno import FNO1d
    from pdelab.training import trainer as trainer_mod

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = TrajectoryDataset(_tiny_dataset(root), NormalizationStats(0.0, 1.0))
        cfg = trainer_mod.TrainConfig(epochs=3, batch_size=2, device="cpu", seed=7)
        torch.manual_seed(3)
        model = FNO1d(modes=4, width=4)
        t = trainer_mod.Trainer(
            model=model, train_set=ds, val_set=ds, stats=NormalizationStats(0.0, 1.0),
            config=cfg, run_id="probe", out_dir=root / "run", system="burgers",
            pde_kind="burgers", pde_params={"nu": 0.01},
        )
        t.train()
        last = torch.load(root / "run" / "last.pt", map_location="cpu", weights_only=False)
        record(
            "resume. last.pt records the current epoch, not the best epoch",
            int(last["epoch"]) == 3,
            f"last.pt['epoch'] = {last['epoch']}, best epoch = {last.get('best_epoch')}",
        )


def probe_resume_is_a_true_continuation() -> None:
    from pdelab.data.dataset import NormalizationStats, TrajectoryDataset
    from pdelab.models.fno import FNO1d
    from pdelab.training import trainer as trainer_mod

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stats = NormalizationStats(0.0, 1.0)
        ds = TrajectoryDataset(_tiny_dataset(root), stats)
        cfg = trainer_mod.TrainConfig(epochs=2, batch_size=2, device="cpu", seed=7)

        def make(out_dir):
            torch.manual_seed(3)
            model = FNO1d(modes=4, width=4)
            return trainer_mod.Trainer(
                model=model, train_set=ds, val_set=ds, stats=stats, config=cfg,
                run_id="probe", out_dir=out_dir, system="burgers",
                pde_kind="burgers", pde_params={"nu": 0.01},
            )

        reference = make(root / "ref")
        reference.train()
        first = make(root / "int")
        try:
            first.train(interrupt_after_batches=2)
        except Exception:
            pass
        second = make(root / "int")
        try:
            second.train(resume_from=root / "int" / "last.pt")
        except TypeError as exc:
            record(
                "resume. an interrupted run can be continued",
                False,
                f"Trainer.train does not accept a checkpoint: {exc}",
            )
            return
        ref = torch.load(root / "ref" / "last.pt", map_location="cpu", weights_only=False)
        res = torch.load(root / "int" / "last.pt", map_location="cpu", weights_only=False)
        same_model = all(
            torch.equal(a, b) for a, b in zip(ref["model"].values(), res["model"].values())
        )
        record(
            "resume. an interrupted run can be continued",
            bool(same_model and ref["global_step"] == res["global_step"]),
            f"same weights = {same_model}, global_step {ref['global_step']} vs "
            f"{res['global_step']}",
        )


def probe_physics_gate_is_hard() -> None:
    from pdelab.data.dataset import NormalizationStats, TrajectoryDataset
    from pdelab.models.base import TimeAdvanceModel
    from pdelab.training import trainer as trainer_mod

    class _Detached(TimeAdvanceModel):
        spatial_dims = 1

        def __init__(self):
            super().__init__(history=1)
            self.scale = torch.nn.Parameter(torch.zeros(1))

        def forward(self, u_hist, params=None):
            return torch.zeros_like(u_hist[:, :1]) + 0.0 * self.scale

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stats = NormalizationStats(0.0, 1.0)
        ds = TrajectoryDataset(_tiny_dataset(root), stats)
        cfg = trainer_mod.TrainConfig(epochs=1, batch_size=2, lambda_pde=0.01,
                                      observation_dt=0.01, device="cpu")
        t = trainer_mod.Trainer(
            model=_Detached(), train_set=ds, val_set=ds, stats=stats, config=cfg,
            run_id="probe", out_dir=root / "run", system="burgers",
            pde_kind="burgers", pde_params={"nu": 0.01},
        )
        try:
            result = t.train()
        except Exception as exc:
            record(
                "training. a zero physics gradient stops the run",
                True,
                f"the run stopped with {type(exc).__name__}: {exc}",
            )
            return
        check = result.physics_grad_check or {}
        record(
            "training. a zero physics gradient stops the run",
            False,
            f"the run completed with physics_grad_check = {check}",
        )


# ----------------------------------------------------------------------
# rollout semantics
# ----------------------------------------------------------------------
def probe_rollout_keeps_the_history_window() -> None:
    from pdelab.data.dataset import NormalizationStats
    from pdelab.evaluation.rollout import rollout
    from pdelab.models.base import TimeAdvanceModel

    class _Recorder(TimeAdvanceModel):
        spatial_dims = 1

        def __init__(self):
            super().__init__(history=2)
            self.inputs = []

        def forward(self, u_hist, params=None):
            self.inputs.append(u_hist.detach().clone())
            return u_hist[:, -1:] * 0.5

    model = _Recorder()
    stats = NormalizationStats(0.0, 1.0)
    hist = torch.arange(2 * 8, dtype=torch.float32).reshape(1, 2, 8)
    try:
        rollout(model, hist, 3, stats, device="cpu")
    except Exception as exc:
        record(
            "rollout. a history of two frames is kept as a sliding window",
            False,
            f"the rollout failed: {type(exc).__name__}: {exc}",
        )
        return
    ok = (
        len(model.inputs) == 3
        and all(w.shape == (1, 2, 8) for w in model.inputs)
        and torch.allclose(model.inputs[1][0, 0], model.inputs[0][0, 1], atol=1e-6)
    )
    record(
        "rollout. a history of two frames is kept as a sliding window",
        bool(ok),
        f"input window shapes = {[tuple(w.shape) for w in model.inputs]}",
    )


def probe_one_step_metric_is_labelled() -> None:
    metrics = importlib.import_module("pdelab.evaluation.metrics")
    summary = metrics.ErrorSummary(
        mean=0.0, median=0.0, std=0.0, min=0.0, max=0.0,
        n_trajectories=1, near_zero_reference_count=0,
    ).to_dict()
    has_aggregation = "aggregation" in summary
    has_helper = hasattr(metrics, "aggregate_windows_per_trajectory")
    record(
        "metrics. the one-step metric says which unit it averages over",
        bool(has_aggregation and has_helper),
        f"aggregation field = {has_aggregation}, per-trajectory helper = {has_helper}",
    )


def probe_numerical_stepping_baseline_exists() -> None:
    metrics = importlib.import_module("pdelab.evaluation.metrics")
    fn = getattr(metrics, "numerical_stepping_forecast", None)
    if fn is None:
        record(
            "baselines. a low-cost numerical stepping baseline exists",
            False,
            "numerical_stepping_forecast is missing",
        )
        return
    x = np.arange(64) / 64
    u0 = (0.5 * np.sin(2 * np.pi * x))[None, :]
    fine = fn(u0, 200, 1e-3 / 20, "burgers", {"nu": 0.01}, scheme="rk2")
    coarse = fn(u0, 10, 1e-3, "burgers", {"nu": 0.01}, scheme="rk2")
    persist = metrics.persistence_forecast(u0, 10)
    reference = fine[:, 19:200:20]
    err_stepped = float(metrics.nrmse_per_trajectory(coarse, reference)[0])
    err_persist = float(metrics.nrmse_per_trajectory(persist, reference)[0])
    record(
        "baselines. a low-cost numerical stepping baseline exists",
        err_stepped < 1e-3 and err_stepped < err_persist,
        f"stepping nRMSE = {err_stepped:.3g}, persistence nRMSE = {err_persist:.3g}",
    )


def main() -> int:
    probe_b_centred_first_derivative_converges()
    probe_b_primary_floor_is_measured()
    probe_b_no_order2_name_survives()
    probe_g_report_does_not_select_the_best_run()
    probe_g_incomplete_cohort_is_never_met()
    probe_provenance_stale_artifact_is_detected()
    probe_provenance_evidence_states_exist()
    probe_resume_records_the_current_epoch()
    probe_resume_is_a_true_continuation()
    probe_physics_gate_is_hard()
    probe_rollout_keeps_the_history_window()
    probe_one_step_metric_is_labelled()
    probe_numerical_stepping_baseline_exists()

    failed = [name for name, ok, _ in RESULTS if not ok]
    print()
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} probes pass")
    if failed:
        print("failing probes:")
        for name in failed:
            print(f"  - {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
