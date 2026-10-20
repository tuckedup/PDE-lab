"""Tests for training, rollout semantics, checkpoint resume and resources."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pdelab.data.dataset import NormalizationStats, TrajectoryDataset
from pdelab.evaluation.rollout import rollout, teacher_forced_one_step
from pdelab.models.fno import FNO1d
from pdelab.resources.preflight import HardwareProfile, JobLock, preflight
from pdelab.training.trainer import TrainConfig, Trainer


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
    preds, refs = teacher_forced_one_step(model, ds, stats, device="cpu")
    assert preds.shape == refs.shape
    assert preds.shape[0] == len(ds)


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
                "config", "rng_state", "normalization", "pde_params"]:
        assert key in ckpt, f"checkpoint is missing {key}"


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
