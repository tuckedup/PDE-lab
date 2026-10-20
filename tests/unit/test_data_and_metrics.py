"""Tests for datasets, splits, normalization and metrics."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pdelab.data.dataset import (
    NormalizationStats,
    TrajectoryDataset,
    compute_normalization_stats,
)
from pdelab.data.generate import restrict_1d, restrict_2d
from pdelab.evaluation.metrics import (
    nrmse_per_snapshot,
    nrmse_per_trajectory,
    persistence_forecast,
    relative_residual_improvement,
    residual_rms,
    summarize_nrmse,
)


# ----------------------------------------------------------------------
# restriction
# ----------------------------------------------------------------------
def test_restrict_1d_preserves_low_modes_exactly():
    """A field that is already band-limited must survive restriction
    unchanged (up to the sampled points)."""
    n_fine, n_obs = 128, 64
    x = np.arange(n_fine) / n_fine
    u = np.cos(2 * np.pi * x)  # only mode +-1
    r = restrict_1d(u, n_obs)
    assert r.shape == (n_obs,)
    xo = np.arange(n_obs) / n_obs
    assert np.allclose(r, np.cos(2 * np.pi * xo), atol=1e-12)


def test_restrict_1d_removes_content_above_the_observation_nyquist():
    """A high-frequency mode must be attenuated, not aliased onto the
    observation grid."""
    n_fine, n_obs = 128, 64
    x = np.arange(n_fine) / n_fine
    u = np.cos(2 * np.pi * 40 * x)  # mode 40, above the observation Nyquist
    r = restrict_1d(u, n_obs)
    # After filtering, the result is (numerically) zero rather than an
    # aliased oscillation.
    assert np.max(np.abs(r)) < 1e-10


def test_restrict_1d_rejects_incompatible_grids():
    with pytest.raises(ValueError):
        restrict_1d(np.zeros(100), 64)


def test_restrict_2d_preserves_low_modes():
    n = 32
    x = np.arange(n) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    r = restrict_2d(u, 16, 16)
    assert r.shape == (16, 16)
    xo = np.arange(16) / 16
    Xo, Yo = np.meshgrid(xo, xo, indexing="ij")
    assert np.allclose(r, np.cos(2 * np.pi * Xo) * np.cos(2 * np.pi * Yo), atol=1e-12)


# ----------------------------------------------------------------------
# splits and normalization
# ----------------------------------------------------------------------
def _write_dataset(path: Path, n_traj: int, n_times: int, n: int, offset: float = 0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fh:
        u = np.zeros((n_traj, n_times, n), dtype=np.float32)
        for t in range(n_traj):
            for s in range(n_times):
                u[t, s] = offset + t + 0.01 * s
        fh.create_dataset("u", data=u)
        fh.create_dataset("ic_seeds", data=np.arange(n_traj, dtype=np.int64))


def test_split_leakage_is_impossible_at_trajectory_level(tmp_path):
    """Windows from one trajectory must never appear in another split."""
    train = tmp_path / "train.h5"
    val = tmp_path / "val.h5"
    test = tmp_path / "test.h5"
    _write_dataset(train, 3, 10, 8, offset=0.0)
    _write_dataset(val, 2, 10, 8, offset=100.0)
    _write_dataset(test, 2, 10, 8, offset=200.0)

    stats = NormalizationStats(mean=0.0, std=1.0)
    tr = TrajectoryDataset(train, stats, history=2)
    va = TrajectoryDataset(val, stats, history=2)
    te = TrajectoryDataset(test, stats, history=2)

    def all_values(ds):
        vals = []
        for i in range(len(ds)):
            item = ds[i]
            vals.append((int(item["traj"]), float(item["history"].sum())))
        return vals

    tr_v = all_values(tr)
    va_v = all_values(va)
    te_v = all_values(te)
    # Distinct offsets guarantee no value can appear in two splits.
    assert not (set(tr_v) & set(va_v))
    assert not (set(tr_v) & set(te_v))
    assert not (set(va_v) & set(te_v))


def test_normalization_is_computed_from_train_only(tmp_path):
    train = tmp_path / "train.h5"
    test = tmp_path / "test.h5"
    _write_dataset(train, 4, 5, 8, offset=0.0)
    _write_dataset(test, 4, 5, 8, offset=1000.0)

    stats = compute_normalization_stats(train)
    train_data = TrajectoryDataset(train, stats).u
    assert stats.mean == pytest.approx(float(train_data.mean()), rel=1e-6)
    # The test split's far larger values must not influence the statistics.
    assert stats.mean < 10.0
    assert stats.std < 10.0


def test_normalization_roundtrip():
    stats = NormalizationStats(mean=1.5, std=2.5)
    x = torch.randn(3, 4)
    assert torch.allclose(stats.denormalize(stats.normalize(x)), x, atol=1e-6)


def test_normalization_from_dict_roundtrip():
    stats = NormalizationStats(mean=0.3, std=1.7)
    assert NormalizationStats.from_dict(stats.to_dict()).mean == 0.3


# ----------------------------------------------------------------------
# metric math
# ----------------------------------------------------------------------
def test_persistence_forecast_tiles_the_time_axis():
    """The persistence forecast must repeat the initial frame along the
    *time* axis.

    Regression test: an inline ``np.repeat(initial[:, None], n, axis=1)``
    is correct, but the same expression applied to a single trajectory
    (where axis 1 is a *spatial* axis) tiles the wrong axis and
    silently produces a meaningless number. The helper removes that
    possibility.
    """
    initial = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    out = persistence_forecast(initial, 5)
    assert out.shape == (2, 5, 4, 4)
    for t in range(5):
        assert np.array_equal(out[:, t], initial)


def test_persistence_forecast_rejects_a_missing_batch_axis():
    with pytest.raises(ValueError):
        persistence_forecast(np.zeros(8), 3)


def test_persistence_nrmse_uses_the_time_axis():
    """A trajectory that drifts linearly must give the expected
    persistence nRMSE; this pins down the axis convention."""
    n_traj, n_times, n = 2, 6, 4
    u = np.zeros((n_traj, n_times, n))
    for t in range(n_times):
        u[:, t] = t  # constant in space, growing in time
    ref = u[:, 1:]
    pred = persistence_forecast(u[:, 0], n_times - 1)
    per_traj = nrmse_per_trajectory(pred, ref)
    # error per step k (k=1..5) is k, reference value is k
    expected = np.sqrt(np.sum([k**2 for k in range(1, n_times)]) /
                       np.sum([k**2 for k in range(1, n_times)]))
    assert np.allclose(per_traj, expected)


def test_nrmse_is_zero_for_a_perfect_prediction():
    ref = np.random.default_rng(0).standard_normal((4, 10, 8))
    assert np.allclose(nrmse_per_trajectory(ref, ref), 0.0, atol=1e-12)


def test_nrmse_matches_the_formula():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((3, 5, 4))
    pred = ref + 0.1 * rng.standard_normal(ref.shape)
    eps = 1e-12
    expected = np.sqrt(
        np.sum((pred - ref) ** 2, axis=(1, 2)) / (np.sum(ref**2, axis=(1, 2)) + eps)
    )
    assert np.allclose(nrmse_per_trajectory(pred, ref), expected)


def test_nrmse_excludes_the_copied_initial_frame():
    """The metric must not include the initial frame, so that copying the
    input cannot deflate the score."""
    rng = np.random.default_rng(2)
    ref = rng.standard_normal((1, 11, 4))
    pred = np.repeat(ref[:, :1], 10, axis=1)  # pure persistence
    value = nrmse_per_trajectory(pred, ref[:, 1:])[0]
    assert value > 0.5


def test_nrmse_handles_a_zero_reference_explicitly():
    ref = np.zeros((2, 4, 4))
    pred = np.ones((2, 4, 4))
    value = nrmse_per_trajectory(pred, ref, epsilon=1e-12)
    assert np.all(np.isfinite(value))
    assert np.all(value > 0)


def test_nrmse_per_snapshot_shape():
    ref = np.random.default_rng(3).standard_normal((4, 6, 8))
    pred = ref + 0.01
    assert nrmse_per_snapshot(pred, ref).shape == (4, 6)


def test_summarize_reports_near_zero_references():
    values = np.array([0.1, 0.2, 0.3])
    energy = np.array([1.0, 0.0, 1e-12])
    s = summarize_nrmse(values, energy)
    assert s.near_zero_reference_count == 2
    assert s.n_trajectories == 3
    assert s.mean == pytest.approx(0.2)


def test_relative_residual_improvement_definition():
    assert relative_residual_improvement(1.0, 0.9) == pytest.approx(10.0)
    assert relative_residual_improvement(1.0, 1.0) == pytest.approx(0.0)
    # A zero baseline denominator must be reported, not silently inflated.
    assert np.isnan(relative_residual_improvement(0.0, 0.5))


def test_residual_rms_aggregation():
    """RMS is taken over space and time per sample, then over samples."""
    r = torch.zeros(2, 3, 4, 4)
    r[0, 0] = 2.0
    r[0, 1] = 2.0
    r[0, 2] = 2.0
    # Sample 0 has a constant residual of 2 over all times and cells;
    # sample 1 is zero.
    value = residual_rms(r)
    expected = np.sqrt((4.0 + 0.0) / 2.0)
    assert value == pytest.approx(expected)


def test_residual_rms_is_scale_invariant_in_shape():
    """The same residual spread over more cells must give the same RMS."""
    a = torch.full((1, 2, 4), 3.0)
    b = torch.full((1, 2, 16), 3.0)
    assert residual_rms(a) == pytest.approx(residual_rms(b))
