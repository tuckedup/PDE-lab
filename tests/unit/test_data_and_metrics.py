"""Tests for datasets, splits, normalization and metrics."""

from __future__ import annotations

import json
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
from pdelab.evaluation.evaluate import (
    residual_rms_for_trajectory,
    residual_temporal_refinement_probe,
    residual_tensor_for_trajectory,
)
from pdelab.evaluation.metrics import (
    aggregate_windows_per_trajectory,
    canonical_json_sha256,
    evaluator_identity,
    nrmse_per_snapshot,
    nrmse_per_trajectory,
    numerical_stepping_forecast,
    persistence_forecast,
    read_digest_sidecar,
    relative_residual_improvement,
    residual_rms,
    sha256_file,
    summarize_nrmse,
    write_json_atomic,
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


# ----------------------------------------------------------------------
# residual evaluators (defect B)
# ----------------------------------------------------------------------
def _exact_uniform_allen_cahn(times: np.ndarray, a: float = 1.0, u0: float = 0.2):
    """The exact spatially uniform Allen-Cahn reaction solution.

    ``u_t = a (u - u^3)`` with ``u(0) = u0``; the Laplacian of a uniform
    field vanishes, so this is an exact solution of the full equation.
    """
    return 1.0 / np.sqrt(1.0 + (1.0 / u0**2 - 1.0) * np.exp(-2.0 * a * times))


def _uniform_trajectory(dt: float) -> np.ndarray:
    times = np.arange(1, int(round(0.2 / dt)) + 1) * dt
    return np.broadcast_to(
        _exact_uniform_allen_cahn(times)[None, :, None, None], (1, len(times), 4, 4)
    ).copy()


def test_centred_first_derivative_residual_converges_at_second_order():
    """The centred *first* derivative must converge at O(dt^2).

    The withdrawn diagnostic used ``(nxt - 2*centre + prv)/dt**2``, which
    is a second time derivative (``u_tt``); it did not converge at all.
    """
    values = []
    for dt in (0.02, 0.01, 0.005):
        value, _info = residual_rms_for_trajectory(
            "allen_cahn",
            _uniform_trajectory(dt),
            dt,
            {"D": 0.001, "a": 1.0},
            "centred_first_derivative",
        )
        values.append(value)
    # Halving dt must quarter the error.
    assert values[0] / values[1] == pytest.approx(4.0, rel=0.05)
    assert values[1] / values[2] == pytest.approx(4.0, rel=0.05)


def test_the_withdrawn_second_time_derivative_does_not_converge():
    """The quantity that used to be reported must not be reportable.

    ``(nxt - 2*centre + prv)/dt**2`` approximates ``u_tt``; on an exact
    solution it stays at a non-zero constant while dt shrinks.
    """
    values = []
    for dt in (0.02, 0.01, 0.005):
        traj = _uniform_trajectory(dt)
        wrong = (traj[:, 2:] - 2 * traj[:, 1:-1] + traj[:, :-2]) / dt**2
        values.append(float(np.sqrt(np.mean(wrong**2))))
    assert values[-1] / values[0] > 0.9  # no convergence
    # ... and it is *not* what the evaluator computes.
    measured, _info = residual_rms_for_trajectory(
        "allen_cahn",
        _uniform_trajectory(0.005),
        0.005,
        {"D": 0.001, "a": 1.0},
        "centred_first_derivative",
    )
    assert measured < values[-1] / 10.0


def test_primary_residual_is_first_order_in_time():
    """The primary (implicit Euler) evaluator is first order in time."""
    values = []
    for dt in (0.02, 0.01, 0.005):
        value, _info = residual_rms_for_trajectory(
            "allen_cahn",
            _uniform_trajectory(dt),
            dt,
            {"D": 0.001, "a": 1.0},
            "implicit_euler_first_order",
        )
        values.append(value)
    assert values[0] / values[1] == pytest.approx(2.0, rel=0.05)
    assert values[1] / values[2] == pytest.approx(2.0, rel=0.05)


def test_both_residual_evaluators_share_the_interior_support():
    """Both evaluators must produce one residual per interior snapshot."""
    traj = np.random.default_rng(0).standard_normal((2, 9, 8)).astype(np.float32)
    for evaluator in ("implicit_euler_first_order", "centred_first_derivative"):
        tensor = residual_tensor_for_trajectory(
            "burgers", traj, 0.01, {"nu": 0.01}, evaluator
        )
        assert tensor.shape == (2, 7, 8)  # T - 2 interior snapshots


def test_temporal_refinement_probe_measures_the_first_order_floor():
    """The floor of the primary evaluator is measured, not asserted."""
    probe = residual_temporal_refinement_probe(
        "allen_cahn",
        {"D": 0.001, "a": 1.0},
        [0.1, 0.05, 0.025],
        horizon=1.0,
        n_spatial=16,
    )
    assert probe["observed_order_implicit_euler_first_order"] == pytest.approx(
        1.0, abs=0.2
    )
    assert probe["observed_order_centred_first_derivative"] == pytest.approx(
        2.0, abs=0.2
    )
    # The primary evaluator's floor at the coarsest dt is finite and
    # reported, so a measured model residual can be compared with it.
    assert probe["rows"][0]["implicit_euler_first_order_truncation_rms"] > 0.0


def test_temporal_refinement_probe_uses_a_fixed_time_window():
    """The probe must compare the same physical window at every dt."""
    probe = residual_temporal_refinement_probe(
        "burgers", {"nu": 0.01}, [0.01, 0.005], horizon=0.5, n_spatial=16
    )
    windows = [(row["n_snapshots"] - 1) * row["dt"] for row in probe["rows"]]
    assert windows[0] == pytest.approx(windows[1], rel=1e-9)


# ----------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------
def test_numerical_stepping_baseline_is_more_accurate_than_persistence():
    """The numerical stepping baseline must actually solve the task."""
    x = np.arange(64) / 64
    u0 = (0.5 * np.sin(2 * np.pi * x))[None, :]
    dt, n_steps = 1e-3, 10
    params = {"nu": 0.01}
    fine = numerical_stepping_forecast(
        u0, n_steps * 20, dt / 20, "burgers", params, scheme="rk2"
    )
    # The fine integration is sampled at the same physical times as the
    # coarse one (the state after step k+1 is at time (k+1) * dt).
    reference = fine[:, 19:200:20]
    stepped = numerical_stepping_forecast(u0, n_steps, dt, "burgers", params, scheme="rk2")
    persist = persistence_forecast(u0, n_steps)

    err_stepped = nrmse_per_trajectory(stepped, reference)[0]
    err_persist = nrmse_per_trajectory(persist, reference)[0]
    assert err_stepped < 1e-3
    assert err_stepped < err_persist


def test_numerical_stepping_baseline_is_parameter_informed():
    """The interface difference from persistence must be real.

    Persistence needs only the initial frame; the numerical baseline
    needs the PDE and its parameters, so it is *not* comparable with a
    learned model in terms of prior knowledge.
    """
    x = np.arange(32) / 32
    u0 = (0.5 * np.sin(2 * np.pi * x))[None, :]
    a = numerical_stepping_forecast(u0, 5, 1e-3, "burgers", {"nu": 0.01})
    b = numerical_stepping_forecast(u0, 5, 1e-3, "burgers", {"nu": 0.05})
    assert not np.allclose(a, b)
    # The persistence forecast cannot depend on any parameter.
    assert np.array_equal(
        persistence_forecast(u0, 5), persistence_forecast(u0, 5)
    )


def test_numerical_stepping_forecast_shares_the_persistence_time_axis():
    """Both baselines must return ``(n_traj, n_steps, *spatial)``."""
    u0 = np.zeros((3, 16), dtype=np.float32)
    stepped = numerical_stepping_forecast(u0, 7, 0.01, "burgers", {"nu": 0.01})
    persist = persistence_forecast(u0, 7)
    assert stepped.shape == persist.shape == (3, 7, 16)


def test_numerical_stepping_forecast_rejects_an_unknown_system():
    with pytest.raises(ValueError):
        numerical_stepping_forecast(np.zeros((2, 8)), 3, 0.01, "not_a_system", {})


# ----------------------------------------------------------------------
# aggregation labels
# ----------------------------------------------------------------------
def test_aggregate_windows_per_trajectory_averages_inside_a_trajectory():
    values = np.array([1.0, 3.0, 10.0, 20.0])
    ids = np.array([0, 0, 1, 1])
    per_traj = aggregate_windows_per_trajectory(values, ids)
    assert np.allclose(per_traj, [2.0, 15.0])


def test_aggregate_windows_per_trajectory_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        aggregate_windows_per_trajectory(np.zeros(3), np.zeros(2))


def test_error_summary_labels_the_aggregation_unit():
    per_window = np.array([1.0, 1.0, 5.0, 5.0])
    per_traj = aggregate_windows_per_trajectory(per_window, np.array([0, 0, 1, 1]))
    window_summary = summarize_nrmse(
        per_window, aggregation="per_window", n_windows=per_window.size
    )
    traj_summary = summarize_nrmse(
        per_traj, aggregation="per_trajectory", n_windows=per_window.size
    )
    assert window_summary.n_trajectories == 4
    assert window_summary.aggregation == "per_window"
    assert traj_summary.n_trajectories == 2
    assert traj_summary.aggregation == "per_trajectory"
    # The per-window mean is not the per-trajectory mean.
    assert window_summary.mean == pytest.approx(3.0)
    assert traj_summary.mean == pytest.approx(3.0)


# ----------------------------------------------------------------------
# content identity
# ----------------------------------------------------------------------
def test_canonical_json_sha256_ignores_key_order():
    a = canonical_json_sha256({"a": 1, "b": [1, 2]})
    b = canonical_json_sha256({"b": [1, 2], "a": 1})
    assert a == b
    assert a != canonical_json_sha256({"a": 2, "b": [1, 2]})


def test_write_json_atomic_hashes_the_finished_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    digest = write_json_atomic(path, {"status": "completed", "value": 1.5})
    assert digest == sha256_file(path)
    assert read_digest_sidecar(path) == digest
    assert json.loads(path.read_text())["value"] == 1.5
    # No temporary file is left behind.
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_json_atomic_replaces_the_previous_content(tmp_path):
    path = tmp_path / "artifact.json"
    write_json_atomic(path, {"value": 1})
    write_json_atomic(path, {"value": 2})
    assert json.loads(path.read_text())["value"] == 2
    assert read_digest_sidecar(path) == sha256_file(path)


def test_digest_sidecar_detects_a_tampered_artifact(tmp_path):
    path = tmp_path / "artifact.json"
    write_json_atomic(path, {"value": 1})
    path.write_text('{"value": 2}')
    assert read_digest_sidecar(path) != sha256_file(path)


def test_evaluator_identity_covers_the_implementation_files():
    identity = evaluator_identity()
    assert identity["id"].startswith("pdelab-evaluation/")
    for rel in (
        "src/pdelab/evaluation/evaluate.py",
        "src/pdelab/evaluation/metrics.py",
        "src/pdelab/evaluation/rollout.py",
        "src/pdelab/physics/operators.py",
    ):
        assert rel in identity["files"]
    assert identity["sha256"] == canonical_json_sha256(identity["files"])
