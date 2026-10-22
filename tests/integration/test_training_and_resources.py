"""Tests for training, rollout semantics, checkpoint resume and resources."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pdelab.data.dataset import NormalizationStats, TrajectoryDataset
from pdelab.evaluation.evaluate import evaluate_run
from pdelab.evaluation.metrics import (
    canonical_json_sha256,
    evaluator_identity,
    sha256_file,
    write_json_atomic,
)
from pdelab.evaluation.rollout import rollout, teacher_forced_one_step
from pdelab.models.base import TimeAdvanceModel
from pdelab.models.fno import FNO1d
from pdelab.reporting.report import (
    build_report,
    build_target_table,
    collect_runs,
)
from pdelab.resources.preflight import HardwareProfile, JobLock, preflight
from pdelab.training.trainer import (
    PhysicsGradientCheckError,
    ResumeMismatchError,
    TrainConfig,
    Trainer,
    TrainingInterrupted,
)


def _make_dataset(path: Path, n_traj: int = 4, n_times: int = 6, n: int = 16):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    with h5py.File(path, "w") as fh:
        u = rng.standard_normal((n_traj, n_times, n)).astype(np.float32)
        # Make it a smooth-in-time sequence so one-step learning is feasible.
        u[:, 1:] = 0.95 * u[:, :-1] + 0.05 * rng.standard_normal((n_traj, n_times - 1, n))
        fh.create_dataset("u", data=u)
        fh.create_dataset("ic_seeds", data=np.arange(n_traj, dtype=np.int64))
    return path


# ----------------------------------------------------------------------
# rollout semantics
# ----------------------------------------------------------------------
def test_rollout_produces_the_requested_number_of_steps():
    model = FNO1d(modes=4, width=4)
    stats = NormalizationStats(mean=0.0, std=1.0)
    hist = torch.randn(2, 1, 16)
    out = rollout(model, hist, 7, stats, device="cpu")
    assert out.shape == (2, 7, 1, 16)


def test_rollout_is_autoregressive():
    """The second step must consume the model's own first prediction, not
    the ground truth."""
    model = FNO1d(modes=4, width=4).eval()
    stats = NormalizationStats(mean=0.0, std=1.0)
    hist = torch.randn(1, 1, 16)
    with torch.no_grad():
        one = rollout(model, hist, 1, stats, device="cpu")
        two = rollout(model, hist, 2, stats, device="cpu")
    assert torch.allclose(two[:, 0], one[:, 0], atol=1e-6)
    assert not torch.allclose(two[:, 1], one[:, 0], atol=1e-6)


def test_teacher_forced_one_step_uses_ground_truth_inputs():
    model = FNO1d(modes=4, width=4).eval()
    stats = NormalizationStats(mean=0.0, std=1.0)
    path = _make_dataset(Path("/tmp/pdelab_test/tf.h5"))
    ds = TrajectoryDataset(path, stats, history=1)
    preds, refs, traj_ids = teacher_forced_one_step(model, ds, stats, device="cpu")
    assert preds.shape == refs.shape
    assert preds.shape[0] == len(ds)
    # The values are per *window*; the trajectory id says which
    # trajectory each window belongs to, so the caller can aggregate
    # them per trajectory instead of mislabelling a per-window mean.
    assert traj_ids.shape == (len(ds),)
    assert len(set(traj_ids.tolist())) < len(ds)  # several windows per trajectory
    assert traj_ids.tolist() == sorted(traj_ids.tolist())


def test_rollout_with_a_longer_history_shifts_the_window():
    """A model with ``history = H`` must receive a *window* of H frames.

    The old rollout replaced the whole history with the single predicted
    frame, which is only correct for ``history == 1``.
    """

    class Recorder(TimeAdvanceModel):
        """A model that records the history windows it is given."""

        spatial_dims = 1

        def __init__(self, history: int = 2):
            super().__init__(history=history)
            self.inputs: list[torch.Tensor] = []

        def forward(self, u_hist, params=None):
            self.inputs.append(u_hist.detach().clone())
            return u_hist[:, -1:] * 0.5

    model = Recorder(history=2)
    stats = NormalizationStats(mean=0.0, std=1.0)
    hist = torch.arange(2 * 8, dtype=torch.float32).reshape(1, 2, 8)
    out = rollout(model, hist, 3, stats, device="cpu")
    assert out.shape == (1, 3, 1, 8)
    assert len(model.inputs) == 3
    # Every input must carry two frames.
    for window in model.inputs:
        assert window.shape == (1, 2, 8)
    # The last frame of the second window is the model's own first
    # prediction, not the original history.
    first_prediction = out[0, 0, 0]
    assert torch.allclose(model.inputs[1][0, -1], first_prediction, atol=1e-6)
    # The first frame is the *oldest* frame of the previous window, so
    # the window really slides instead of being replaced.
    assert torch.allclose(model.inputs[1][0, 0], model.inputs[0][0, 1], atol=1e-6)


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
def test_physics_term_produces_nonzero_finite_gradients(tmp_path):
    """The PDE regulariser must actually constrain the model."""
    path = _make_dataset(tmp_path / "train.h5")
    stats = NormalizationStats(mean=0.0, std=1.0)
    ds = TrajectoryDataset(path, stats, history=1)
    model = FNO1d(modes=4, width=4)
    cfg = TrainConfig(epochs=1, batch_size=2, lr=1e-3, lambda_pde=0.01,
                      observation_dt=0.01, seed=0, device="cpu")
    trainer = Trainer(
        model=model, train_set=ds, val_set=ds, stats=stats, config=cfg,
        run_id="test", out_dir=tmp_path / "run", system="burgers",
        pde_kind="burgers", pde_params={"nu": 0.01},
    )
    result = trainer.train()
    check = result.physics_grad_check
    assert check is not None
    assert check["grads_finite"] is True
    assert check["grads_nonzero"] is True
    assert check["passed"] is True
    assert check["grad_l1_sum"] > 0


def test_data_only_condition_reports_no_physics_check(tmp_path):
    path = _make_dataset(tmp_path / "train.h5")
    stats = NormalizationStats(mean=0.0, std=1.0)
    ds = TrajectoryDataset(path, stats, history=1)
    model = FNO1d(modes=4, width=4)
    cfg = TrainConfig(epochs=1, batch_size=2, device="cpu")
    trainer = Trainer(
        model=model, train_set=ds, val_set=ds, stats=stats, config=cfg,
        run_id="test", out_dir=tmp_path / "run", system="burgers",
        pde_kind="burgers", pde_params={"nu": 0.01},
    )
    result = trainer.train()
    assert result.physics_grad_check is None


class _DetachedModel(TimeAdvanceModel):
    """A model whose output does not depend on its parameters.

    The physics term therefore cannot produce a gradient, which is
    exactly the failure the gate has to catch.
    """

    spatial_dims = 1

    def __init__(self, value: float = 0.0, non_finite: bool = False):
        super().__init__(history=1)
        self.value = value
        self.non_finite = non_finite
        self.scale = torch.nn.Parameter(torch.zeros(1))

    def forward(self, u_hist, params=None):
        out = torch.full(
            (u_hist.shape[0], 1, u_hist.shape[-1]),
            float("nan") if self.non_finite else self.value,
            dtype=u_hist.dtype,
        )
        return out + 0.0 * self.scale  # keeps the parameter in the graph


def test_physics_gradient_gate_stops_a_run_with_a_zero_gradient(tmp_path):
    """A zero physics gradient is a hard failure, not a recorded note."""
    path = _make_dataset(tmp_path / "train.h5")
    stats = NormalizationStats(mean=0.0, std=1.0)
    ds = TrajectoryDataset(path, stats, history=1)
    model = _DetachedModel()
    cfg = TrainConfig(epochs=1, batch_size=2, lr=1e-3, lambda_pde=0.01,
                      observation_dt=0.01, seed=0, device="cpu")
    trainer = Trainer(
        model=model, train_set=ds, val_set=ds, stats=stats, config=cfg,
        run_id="test", out_dir=tmp_path / "run", system="burgers",
        pde_kind="burgers", pde_params={"nu": 0.01},
    )
    with pytest.raises(PhysicsGradientCheckError) as excinfo:
        trainer.train()
    assert "no usable gradient" in str(excinfo.value)
    check = getattr(excinfo.value, "physics_grad_check", None)
    assert check is not None
    assert check["passed"] is False


def test_physics_gradient_gate_stops_a_run_with_a_non_finite_gradient(tmp_path):
    path = _make_dataset(tmp_path / "train.h5")
    stats = NormalizationStats(mean=0.0, std=1.0)
    ds = TrajectoryDataset(path, stats, history=1)
    model = _DetachedModel(non_finite=True)
    cfg = TrainConfig(epochs=1, batch_size=2, lr=1e-3, lambda_pde=0.01,
                      observation_dt=0.01, seed=0, device="cpu")
    trainer = Trainer(
        model=model, train_set=ds, val_set=ds, stats=stats, config=cfg,
        run_id="test", out_dir=tmp_path / "run", system="burgers",
        pde_kind="burgers", pde_params={"nu": 0.01},
    )
    with pytest.raises(PhysicsGradientCheckError):
        trainer.train()


def test_checkpoint_contains_resume_state(tmp_path):
    path = _make_dataset(tmp_path / "train.h5")
    stats = NormalizationStats(mean=0.0, std=1.0)
    ds = TrajectoryDataset(path, stats, history=1)
    model = FNO1d(modes=4, width=4)
    cfg = TrainConfig(epochs=1, batch_size=2, device="cpu", seed=7)
    trainer = Trainer(
        model=model, train_set=ds, val_set=ds, stats=stats, config=cfg,
        run_id="test", out_dir=tmp_path / "run", system="burgers",
        pde_kind="burgers", pde_params={"nu": 0.01},
    )
    trainer.train()
    ckpt = torch.load(tmp_path / "run" / "best.pt", map_location="cpu", weights_only=False)
    for key in ["model", "optimizer", "scheduler", "epoch", "global_step",
                "config", "rng_state", "normalization", "pde_params",
                "numpy_rng_state", "python_rng_state", "cuda_rng_state",
                "shuffle_generator_state", "epoch_batches", "batch_index",
                "epochs_completed", "data_identity", "stopping_policy",
                "data_position", "resume_schema_version"]:
        assert key in ckpt, f"checkpoint is missing {key}"


def test_last_checkpoint_records_the_current_epoch_not_the_best_epoch(tmp_path):
    """``last.pt`` must say where the loop actually is."""
    path = _make_dataset(tmp_path / "train.h5")
    stats = NormalizationStats(mean=0.0, std=1.0)
    ds = TrajectoryDataset(path, stats, history=1)
    torch.manual_seed(3)
    model = FNO1d(modes=4, width=4)
    cfg = TrainConfig(epochs=3, batch_size=2, device="cpu", seed=7)
    trainer = Trainer(
        model=model, train_set=ds, val_set=ds, stats=stats, config=cfg,
        run_id="test", out_dir=tmp_path / "run", system="burgers",
        pde_kind="burgers", pde_params={"nu": 0.01},
    )
    result = trainer.train()
    last = torch.load(tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(tmp_path / "run" / "best.pt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 3  # three epochs were completed
    assert last["epochs_completed"] == 3
    assert last["last_executed_epoch"] == 2
    assert last["batch_index"] == 0
    assert result.last_epoch == 2
    # The best epoch is recorded separately and is *not* the current epoch.
    assert last["best_epoch"] == best["best_epoch"] == result.best_epoch
    assert last["epoch"] != last["best_epoch"]


def _assert_same(a, b, path="state") -> None:
    """Recursive equality for nested checkpoint payloads."""
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"{path} differs"
    elif isinstance(a, dict):
        # The wall-clock timings are not state: a resumed run has spent
        # its time in two processes.
        a = {k: v for k, v in a.items() if k != "elapsed_seconds"}
        b = {k: v for k, v in b.items() if k != "elapsed_seconds"}
        assert set(a) == set(b), f"{path} keys differ: {set(a) ^ set(b)}"
        for key in a:
            _assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path} length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), f"{path} differs"
    else:
        assert a == b, f"{path} differs: {a!r} != {b!r}"


def test_interrupted_and_resumed_run_ends_in_the_same_state(tmp_path):
    """An interrupted-and-resumed run must end where an uninterrupted one ends.

    Checking that the checkpoint *has* the keys is not enough: the
    weights, the optimizer and scheduler state, the data order, the RNG
    states and the loop position must all agree.
    """
    path = _make_dataset(tmp_path / "train.h5")
    stats = NormalizationStats(mean=0.0, std=1.0)
    ds = TrajectoryDataset(path, stats, history=1)
    cfg = TrainConfig(epochs=4, batch_size=2, device="cpu", seed=7, warmup_steps=2)
    data_identity = {"dataset_dir": str(tmp_path), "splits": {"train": {"sha256": "x"}}}

    def make_trainer(out_dir):
        torch.manual_seed(3)
        model = FNO1d(modes=4, width=4)
        return Trainer(
            model=model, train_set=ds, val_set=ds, stats=stats, config=cfg,
            run_id="test", out_dir=out_dir, system="burgers",
            pde_kind="burgers", pde_params={"nu": 0.01},
            data_identity=data_identity,
        )

    # (1) the uninterrupted reference run
    reference_dir = tmp_path / "reference"
    reference = make_trainer(reference_dir)
    ref_result = reference.train()

    # (2) the same run, interrupted in the middle of the first epoch
    interrupted_dir = tmp_path / "interrupted"
    first = make_trainer(interrupted_dir)
    with pytest.raises(TrainingInterrupted):
        first.train(interrupt_after_batches=3)

    # (3) ... and resumed
    second = make_trainer(interrupted_dir)
    resumed = second.train(resume_from=interrupted_dir / "last.pt")

    ref_ckpt = torch.load(reference_dir / "last.pt", map_location="cpu", weights_only=False)
    res_ckpt = torch.load(interrupted_dir / "last.pt", map_location="cpu", weights_only=False)

    for key in (
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "epochs_completed",
        "batch_index",
        "global_step",
        "shuffle_generator_state",
        "rng_state",
        "numpy_rng_state",
        "python_rng_state",
        "best_val_loss",
        "best_epoch",
        "history",
    ):
        _assert_same(ref_ckpt[key], res_ckpt[key], key)

    assert resumed.epochs_completed == ref_result.epochs_completed == 4
    assert resumed.global_step == ref_result.global_step
    assert resumed.best_val_loss == pytest.approx(ref_result.best_val_loss, rel=1e-12)
    # The per-epoch history must agree, which also proves that the
    # resumed run saw the same data in the same order.
    assert len(resumed.history) == len(ref_result.history)
    for a, b in zip(ref_result.history, resumed.history):
        assert a["epoch"] == b["epoch"]
        assert a["train"]["total"] == pytest.approx(b["train"]["total"], rel=1e-12)
        assert a["val"]["total"] == pytest.approx(b["val"]["total"], rel=1e-12)


def test_resume_refuses_a_checkpoint_from_other_data(tmp_path):
    path = _make_dataset(tmp_path / "train.h5")
    stats = NormalizationStats(mean=0.0, std=1.0)
    ds = TrajectoryDataset(path, stats, history=1)
    cfg = TrainConfig(epochs=1, batch_size=2, device="cpu", seed=7)
    torch.manual_seed(3)
    model = FNO1d(modes=4, width=4)
    trainer = Trainer(
        model=model, train_set=ds, val_set=ds, stats=stats, config=cfg,
        run_id="test", out_dir=tmp_path / "run", system="burgers",
        pde_kind="burgers", pde_params={"nu": 0.01},
        data_identity={"dataset_dir": "a"},
    )
    trainer.train()

    torch.manual_seed(3)
    other = FNO1d(modes=4, width=4)
    other_trainer = Trainer(
        model=other, train_set=ds, val_set=ds, stats=stats, config=cfg,
        run_id="test", out_dir=tmp_path / "run2", system="burgers",
        pde_kind="burgers", pde_params={"nu": 0.01},
        data_identity={"dataset_dir": "b"},
    )
    with pytest.raises(ResumeMismatchError):
        other_trainer.train(resume_from=tmp_path / "run" / "last.pt")


def test_resume_restores_the_same_weights(tmp_path):
    path = _make_dataset(tmp_path / "train.h5")
    stats = NormalizationStats(mean=0.0, std=1.0)
    ds = TrajectoryDataset(path, stats, history=1)
    model = FNO1d(modes=4, width=4)
    cfg = TrainConfig(epochs=1, batch_size=2, device="cpu", seed=7)
    trainer = Trainer(
        model=model, train_set=ds, val_set=ds, stats=stats, config=cfg,
        run_id="test", out_dir=tmp_path / "run", system="burgers",
        pde_kind="burgers", pde_params={"nu": 0.01},
    )
    trainer.train()
    ckpt = torch.load(tmp_path / "run" / "best.pt", map_location="cpu", weights_only=False)
    restored = FNO1d(modes=4, width=4)
    restored.load_state_dict(ckpt["model"])
    for p, q in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(p, q)


def test_seeded_initialisation_is_deterministic():
    """The model initialisation must be a deterministic function of the
    seed.

    Regression test: the seed used to be applied inside the trainer,
    which runs *after* the model is constructed, so the initialisation
    was at the mercy of the process-level RNG and a seeded run was not
    reproducible.
    """
    from pdelab.utils import init_fingerprint, seed_everything

    def fingerprint(seed: int) -> str:
        seed_everything(seed)
        model = FNO1d(modes=4, width=4)
        return init_fingerprint(model)

    assert fingerprint(0) == fingerprint(0)
    assert fingerprint(0) != fingerprint(1)


def test_seed_everything_is_idempotent():
    from pdelab.utils import seed_everything

    seed_everything(3)
    a = torch.randn(5)
    seed_everything(3)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_training_is_reproducible_for_a_fixed_seed(tmp_path):
    path = _make_dataset(tmp_path / "train.h5")
    stats = NormalizationStats(mean=0.0, std=1.0)
    ds = TrajectoryDataset(path, stats, history=1)
    losses = []
    for _ in range(2):
        # The seed must be set before the model is constructed, because
        # the parameter initialisation consumes the global RNG.
        torch.manual_seed(11)
        np.random.seed(11)
        model = FNO1d(modes=4, width=4)
        cfg = TrainConfig(epochs=2, batch_size=2, device="cpu", seed=11)
        trainer = Trainer(
            model=model, train_set=ds, val_set=ds, stats=stats, config=cfg,
            run_id="test", out_dir=tmp_path / "run", system="burgers",
            pde_kind="burgers", pde_params={"nu": 0.01},
        )
        result = trainer.train()
        losses.append(result.best_val_loss)
    assert losses[0] == pytest.approx(losses[1], rel=1e-9)


# ----------------------------------------------------------------------
# evaluation artifacts and the report's cohort model
# ----------------------------------------------------------------------
def _make_run_dir(
    tmp_path,
    *,
    run_id: str,
    runs_root: Path | None = None,
    system: str = "burgers",
    architecture: str = "fno1d",
    condition: str = "data_only",
    seed: int = 0,
    lambda_pde: float = 0.0,
    n_traj: int = 4,
    n_times: int = 8,
    n: int = 32,
    dt: float = 0.01,
    history: int = 1,
):
    """A minimal run directory: dataset split, manifest and checkpoint."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    h5 = data_dir / f"{system}_test.h5"
    rng = np.random.default_rng(3)
    with h5py.File(h5, "w") as fh:
        u = rng.standard_normal((n_traj, n_times, n)).astype(np.float32)
        u[:, 1:] = 0.95 * u[:, :-1] + 0.05 * rng.standard_normal((n_traj, n_times - 1, n))
        fh.create_dataset("u", data=u)
        fh.create_dataset("ic_seeds", data=np.arange(n_traj))

    root = Path(runs_root) if runs_root else tmp_path / "runs"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model = FNO1d(modes=4, width=4, history=history)
    torch.save(
        {
            "model": model.state_dict(),
            "normalization": NormalizationStats(mean=0.0, std=1.0).to_dict(),
            "epoch": 0,
        },
        run_dir / "best.pt",
    )
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "experiment": {
            "system": system,
            "architecture": architecture,
            "condition": condition,
            "seed": seed,
            "lambda_pde": lambda_pde,
            "history": history,
            "dataset_dir": str(data_dir),
            "model_kwargs": {"modes": 4, "width": 4},
        },
        "dataset_manifest": {
            "system": {
                "observation_times": [i * dt for i in range(n_times)],
                "params": {"nu": 0.01},
            },
            "files": {"test": {"sha256": sha256_file(h5)}},
        },
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    return run_dir, h5


def _write_evaluation_artifact(
    run_dir: Path,
    h5: Path,
    *,
    nrmse: float,
    residual: float = 1.0,
    reference_residual: float = 1.2,
    evaluator: dict | None = None,
    with_identity: bool = True,
    corrupt: bool = False,
):
    """Write an evaluation artifact for a fixture run."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    identity = {
        "evaluator": evaluator or evaluator_identity(),
        "protocol": {"id": "pdelab-free-rollout/2", "sha256": "proto"},
        "resolved_settings": {"split": "test"},
        "resolved_settings_sha256": canonical_json_sha256({"split": "test"}),
        "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "dataset": {
            "split": "test",
            "path": str(h5),
            "sha256": sha256_file(h5),
        },
        "normalization": {"mean": 0.0, "std": 1.0, "source": "train"},
    }
    artifact = {
        "status": "completed",
        "run_id": manifest["run_id"],
        "split": "test",
        "system": manifest["experiment"]["system"],
        "architecture": manifest["experiment"]["architecture"],
        "condition": manifest["experiment"]["condition"],
        "seed": manifest["experiment"]["seed"],
        "checkpoint": str(run_dir / "best.pt"),
        "test_dataset_sha256": sha256_file(h5),
        "identity": identity,
        "protocol": {"id": "pdelab-free-rollout/2", "split": "test"},
        "nrmse": {"mean": nrmse, "median": nrmse, "std": 0.0, "min": nrmse,
                  "max": nrmse, "n_trajectories": 4, "near_zero_reference_count": 0},
        "residual_rms_implicit_euler_first_order": residual,
        "reference_residual_rms_implicit_euler_first_order": reference_residual,
        "residual_rms_centred_first_derivative": residual / 10.0,
        "reference_residual_rms_centred_first_derivative": reference_residual / 10.0,
    }
    if not with_identity:
        artifact.pop("identity")
    path = run_dir / "evaluation.json"
    if corrupt:
        path.write_text("{not json")
    else:
        write_json_atomic(path, artifact)
    return path


def _cohort(run_ids, *, dataset_sha256=None, evaluator=None, seeds=(0,)):
    """A cohort declaration for the fixture runs."""
    members = []
    for run_id in run_ids:
        members.append({
            "id": run_id,
            "system": "burgers",
            "architecture": "fno1d",
            "condition": "data_only",
            "lambda_pde": 0.0,
            "seeds": list(seeds),
            "run_ids": [run_id],
        })
    return {
        "schema": "pdelab-evidence-cohort/1",
        "cohort_id": "test-cohort",
        "evaluator": evaluator or evaluator_identity(),
        "protocol": {"id": "pdelab-free-rollout/2", "split": "test"},
        "datasets": {
            "burgers": {"test_sha256": dataset_sha256},
        },
        "members": members,
        "targets": [
            {
                "name": "burgers_nrmse",
                "metric": "nrmse",
                "members": list(run_ids),
            }
        ],
    }


TARGETS = {
    "targets": [
        {
            "name": "burgers_nrmse",
            "metric": "nrmse",
            "threshold": 0.05,
            "comparison": "<",
            "units": "dimensionless",
        }
    ]
}


def test_report_evaluates_a_target_against_the_declared_cohort(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    _write_evaluation_artifact(run_dir, h5, nrmse=0.01)
    runs = collect_runs(tmp_path / "runs")
    cohort = _cohort(["run0"], dataset_sha256=sha256_file(h5))
    rows = build_target_table(runs, TARGETS, cohort)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "met"
    assert row["measured"] == pytest.approx(0.01)
    assert row["run_id"] == "run0"


def test_report_does_not_select_the_best_test_run(tmp_path):
    """The report must not pick the most favourable run of the cohort.

    Two runs are declared; one is far better than the other.  The
    target is met only if *every* declared member satisfies it, and the
    reported value is the mean over the members, not the best of them.
    """
    run_a, h5 = _make_run_dir(tmp_path, run_id="run_a")
    _write_evaluation_artifact(run_a, h5, nrmse=0.01)
    run_b, _ = _make_run_dir(tmp_path, run_id="run_b")
    _write_evaluation_artifact(run_b, h5, nrmse=0.20)
    runs = collect_runs(tmp_path / "runs")
    cohort = _cohort(["run_a", "run_b"], dataset_sha256=sha256_file(h5))
    rows = build_target_table(runs, TARGETS, cohort)
    row = rows[0]
    assert row["status"] == "missed"
    assert row["measured"] == pytest.approx(0.105)
    assert "run_a=0.01" in row["note"] and "run_b=0.2" in row["note"]


def test_report_reports_stale_for_an_artifact_older_than_the_checkpoint(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    path = _write_evaluation_artifact(run_dir, h5, nrmse=0.01)
    # The checkpoint is replaced after the evaluation was produced.
    later = time.time() + 60
    os.utime(run_dir / "best.pt", (later, later))
    runs = collect_runs(tmp_path / "runs")
    cohort = _cohort(["run0"], dataset_sha256=sha256_file(h5))
    rows = build_target_table(runs, TARGETS, cohort)
    row = rows[0]
    assert row["status"] == "stale"
    assert row["measured"] is None
    assert "older than the checkpoint" in row["note"]
    assert path.exists()


def test_report_reports_stale_for_a_replaced_checkpoint(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    _write_evaluation_artifact(run_dir, h5, nrmse=0.01)
    # The checkpoint bytes change but the file keeps its (newer) time.
    torch.save({"model": {}, "normalization": {}}, run_dir / "best.pt")
    later = time.time() + 60
    os.utime(run_dir / "best.pt", (later, later))
    runs = collect_runs(tmp_path / "runs")
    cohort = _cohort(["run0"], dataset_sha256=sha256_file(h5))
    rows = build_target_table(runs, TARGETS, cohort)
    assert rows[0]["status"] == "stale"
    assert rows[0]["measured"] is None


def test_report_reports_invalid_protocol_for_an_artifact_without_identity(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    _write_evaluation_artifact(run_dir, h5, nrmse=0.01, with_identity=False)
    runs = collect_runs(tmp_path / "runs")
    cohort = _cohort(["run0"], dataset_sha256=sha256_file(h5))
    rows = build_target_table(runs, TARGETS, cohort)
    assert rows[0]["status"] == "invalid_protocol"
    assert rows[0]["measured"] is None
    assert "content identity" in rows[0]["note"]


def test_report_reports_invalid_protocol_for_a_foreign_evaluator(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    foreign = {
        "id": "pdelab-evaluation/1",
        "sha256": "0" * 64,
        "files": {"src/pdelab/evaluation/evaluate.py": "0" * 64},
    }
    _write_evaluation_artifact(run_dir, h5, nrmse=0.01, evaluator=foreign)
    runs = collect_runs(tmp_path / "runs")
    cohort = _cohort(["run0"], dataset_sha256=sha256_file(h5))
    rows = build_target_table(runs, TARGETS, cohort)
    assert rows[0]["status"] == "invalid_protocol"
    assert rows[0]["measured"] is None
    assert "evaluator implementation has changed" in rows[0]["note"]


def test_report_reports_failed_for_a_corrupt_artifact(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    _write_evaluation_artifact(run_dir, h5, nrmse=0.01, corrupt=True)
    runs = collect_runs(tmp_path / "runs")
    cohort = _cohort(["run0"], dataset_sha256=sha256_file(h5))
    rows = build_target_table(runs, TARGETS, cohort)
    assert rows[0]["status"] == "failed"
    assert rows[0]["measured"] is None
    assert "corrupt" in rows[0]["note"]


def test_report_reports_incomplete_cohort_never_met(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    _write_evaluation_artifact(run_dir, h5, nrmse=0.01)
    runs = collect_runs(tmp_path / "runs")
    # The cohort declares a second run that was never evaluated.
    cohort = _cohort(["run0", "run1"], dataset_sha256=sha256_file(h5))
    rows = build_target_table(runs, TARGETS, cohort)
    row = rows[0]
    assert row["status"] in ("not_run", "incomplete_cohort")
    assert row["status"] != "met"
    assert row["measured"] is None
    assert "run1" in row["note"]


def test_report_excludes_a_run_that_does_not_match_the_cohort_identity(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0", condition="physics")
    _write_evaluation_artifact(run_dir, h5, nrmse=0.01)
    runs = collect_runs(tmp_path / "runs")
    cohort = _cohort(["run0"], dataset_sha256=sha256_file(h5))
    rows = build_target_table(runs, TARGETS, cohort)
    assert rows[0]["status"] == "invalid_protocol"
    assert rows[0]["measured"] is None
    assert "does not match the cohort member" in rows[0]["note"]


def test_report_without_a_cohort_never_reports_met(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    _write_evaluation_artifact(run_dir, h5, nrmse=0.01)
    runs = collect_runs(tmp_path / "runs")
    rows = build_target_table(runs, TARGETS, {})
    assert rows[0]["status"] == "not_run"
    assert rows[0]["measured"] is None
    assert "cohort" in rows[0]["note"]


def test_report_does_not_substitute_the_best_benchmark(tmp_path):
    """A missing declared reference must fail explicitly."""
    ref_dir, h5 = _make_run_dir(tmp_path, run_id="reference")
    other_dir, _ = _make_run_dir(tmp_path, run_id="other")
    write_json_atomic(
        other_dir / "benchmark_matched.json",
        {"status": "completed", "run_id": "other", "throughput_ratio": 99.0},
    )
    runs = collect_runs(tmp_path / "runs")
    cohort = {
        "cohort_id": "test-cohort",
        "evaluator": evaluator_identity(),
        "protocol": {"id": "pdelab-free-rollout/2"},
        "members": [],
        "targets": [
            {"name": "inference_throughput_ratio", "metric": "throughput_ratio",
             "reference": "reference"}
        ],
    }
    targets = {
        "targets": [
            {"name": "inference_throughput_ratio", "metric": "throughput_ratio",
             "threshold": 1.3, "comparison": ">=", "units": "ratio"}
        ],
        "inference_reference_model": "reference",
    }
    rows = build_target_table(runs, targets, cohort)
    row = rows[0]
    assert row["status"] != "met"
    assert row["measured"] is None
    assert "reference" in row["note"]


def test_build_report_on_a_stale_artifact_set_reports_a_state_and_no_number(tmp_path):
    """The acceptance case: a stale artifact set yields a state, not a number."""
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    _write_evaluation_artifact(run_dir, h5, nrmse=0.01)
    later = time.time() + 60
    os.utime(run_dir / "best.pt", (later, later))
    cohort_path = tmp_path / "cohort.yaml"
    cohort_path.write_text(
        "schema: pdelab-evidence-cohort/1\n"
        "cohort_id: test-cohort\n"
        "evaluator:\n"
        f"  id: pdelab-evaluation/2\n"
        f"  sha256: {evaluator_identity()['sha256']}\n"
        "protocol:\n"
        "  id: pdelab-free-rollout/2\n"
        "members:\n"
        "  - id: run0\n"
        "    system: burgers\n"
        "    architecture: fno1d\n"
        "    condition: data_only\n"
        "    seeds: [0]\n"
        "    run_ids: [run0]\n"
        "targets:\n"
        "  - name: burgers_nrmse\n"
        "    metric: nrmse\n"
        "    members: [run0]\n"
    )
    out = build_report(
        output_dir=tmp_path / "results",
        make_plots=False,
        runs_dir=tmp_path / "runs",
        cohort_path=cohort_path,
        write_docs=False,
    )
    text = (tmp_path / "results" / "RESULTS.md").read_text()
    assert "**stale**" in text
    assert out["targets_met"] == 0
    payload = json.loads((tmp_path / "results" / "results.json").read_text())
    row = payload["targets"][0]
    assert row["status"] == "stale"
    assert row["measured"] is None
    assert "older than the checkpoint" in row["note"]


def test_evaluation_artifact_binds_the_content_identity(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    result = evaluate_run(run_dir, device="cpu", split="test")
    assert result["status"] == "completed"
    artifact = json.loads((run_dir / "evaluation.json").read_text())
    identity = artifact["identity"]
    assert identity["checkpoint_sha256"] == sha256_file(run_dir / "best.pt")
    assert identity["dataset"]["sha256"] == sha256_file(h5)
    assert identity["evaluator"]["sha256"] == evaluator_identity()["sha256"]
    assert identity["resolved_settings_sha256"]
    assert identity["protocol"]["id"] == "pdelab-free-rollout/2"
    assert identity["normalization"]["sha256"]
    # The artifact is written atomically and its digest is recorded next
    # to it.
    sidecar = (run_dir / "evaluation.json.sha256").read_text().strip()
    assert sidecar == sha256_file(run_dir / "evaluation.json")


def test_evaluation_reports_the_reference_residual_under_every_evaluator(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    result = evaluate_run(run_dir, device="cpu", split="test")
    assert result["status"] == "completed"
    artifact = json.loads((run_dir / "evaluation.json").read_text())
    for evaluator in ("implicit_euler_first_order", "centred_first_derivative"):
        assert f"residual_rms_{evaluator}" in artifact
        assert f"reference_residual_rms_{evaluator}" in artifact
        assert artifact["residual_evaluators"][evaluator]["support"] == (
            "interior_snapshots_only"
        )
    # The withdrawn "order2" identity must not survive anywhere.
    assert "residual_error_physical_rms_order2" not in artifact
    assert "reference_residual_physical_rms_order2" not in artifact
    # The floor of the primary evaluator is measured, not asserted.
    probe = artifact["residual_temporal_refinement_probe"]
    assert probe["observed_order_implicit_euler_first_order"] == pytest.approx(
        1.0, abs=0.3
    )
    assert probe["observed_order_centred_first_derivative"] == pytest.approx(
        2.0, abs=0.3
    )


def test_evaluation_reports_the_numerical_stepping_baseline(tmp_path):
    run_dir, h5 = _make_run_dir(tmp_path, run_id="run0")
    evaluate_run(run_dir, device="cpu", split="test")
    artifact = json.loads((run_dir / "evaluation.json").read_text())
    assert "numerical_stepping_baseline_nrmse" in artifact
    assert "numerical_stepping_baseline_walltime_seconds" in artifact
    assert "persistence_baseline_walltime_seconds" in artifact
    # The one-step metric is aggregated per trajectory, and the per
    # window summary is reported separately.
    assert artifact["nrmse_one_step_teacher_forced"]["aggregation"] == "per_trajectory"
    assert artifact["nrmse_one_step_teacher_forced"]["n_trajectories"] == 4
    assert artifact["nrmse_one_step_teacher_forced_per_window"]["aggregation"] == (
        "per_window"
    )


# ----------------------------------------------------------------------
# resources
# ----------------------------------------------------------------------
def test_hardware_profile_loads_with_expected_fields():
    profile = HardwareProfile.load("configs/hardware/atul_laptop.yaml")
    assert profile.experiment_concurrency == 1
    assert profile.cpu_threads <= 4
    assert profile.dataloader_workers == 0
    assert profile.training_seeds_confirmatory == [0, 1, 2]
    assert profile.run_full_sweep_by_default is False


def test_preflight_reports_a_structured_result():
    profile = HardwareProfile.load("configs/hardware/atul_laptop.yaml")
    report = preflight(profile, require_gpu=False)
    assert "violations" in report
    assert "warnings" in report
    assert "host" in report
    assert "gpu" in report
    assert isinstance(report["ok"], bool)


def test_job_lock_is_exclusive():
    lock_a = JobLock(Path("/tmp/pdelab_test_lock"))
    lock_b = JobLock(Path("/tmp/pdelab_test_lock"))
    assert lock_a.acquire(blocking=False)
    try:
        assert not lock_b.acquire(blocking=False)
    finally:
        lock_a.release()
    assert lock_b.acquire(blocking=False)
    lock_b.release()
