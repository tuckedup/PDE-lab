"""Fixture: the inference benchmark's accuracy check must reject a sign
flip and a spatial permutation.

The old check compared the *scalar* mean absolute value of the rollout
of the baseline and of the optimized variant.  Two completely different
trajectories can share that number: a sign flip and a spatial
permutation both leave it unchanged, so both passed the old check.

The repaired check compares the **full output tensors** element by
element (max absolute error, relative L2, non-finite counts, the
difference at every time step and the change of the rollout nRMSE), so
both are rejected.

Run it either way::

    .venv/bin/python -m pytest campaigns/performance/test_benchmark_correctness.py -q
    .venv/bin/python campaigns/performance/test_benchmark_correctness.py
"""

from __future__ import annotations

import numpy as np

from pdelab.benchmarks.inference import (
    ACCURACY_TOLERANCE,
    compare_rollout_outputs,
    legacy_checksum_agreement,
    legacy_scalar_checksum,
)


def _rollout(seed: int = 11, n_traj: int = 4, n_steps: int = 6, n: int = 16) -> np.ndarray:
    """A smooth, non-trivial rollout of shape ``(n_traj, n_steps, n, n)``."""
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((n_traj, n_steps, n, n)).astype(np.float64)
    # smooth in space and time so the field is not dominated by noise
    for _ in range(3):
        field = 0.5 * field + 0.5 * np.roll(field, 1, axis=-1)
        field = 0.5 * field + 0.5 * np.roll(field, 1, axis=-2)
    return field


def _sign_flip(baseline: np.ndarray) -> np.ndarray:
    return -baseline


def _spatial_permutation(baseline: np.ndarray) -> np.ndarray:
    """A permutation of the spatial samples (the mean abs value is kept)."""
    return np.roll(baseline, 5, axis=-1)


def _reference(n_traj: int, n_steps: int, n: int) -> np.ndarray:
    """A stand-in for the ground truth, used only for the nRMSE delta."""
    return _rollout(seed=23, n_traj=n_traj, n_steps=n_steps, n=n) * 1.05


# ----------------------------------------------------------------------
# the checks
# ----------------------------------------------------------------------
def test_the_old_scalar_checksum_accepts_a_sign_flip():
    """Documents the defect: the checksum cannot see a sign flip."""
    baseline = _rollout()
    flipped = _sign_flip(baseline)
    assert legacy_scalar_checksum(baseline) == legacy_scalar_checksum(flipped)
    legacy = legacy_checksum_agreement(baseline, flipped)
    assert legacy["within_tolerance"], (
        "the legacy checksum check is expected to accept the sign flip - that "
        "is the defect this fixture documents"
    )


def test_the_old_scalar_checksum_accepts_a_spatial_permutation():
    """Documents the defect: the checksum cannot see a permutation."""
    baseline = _rollout()
    permuted = _spatial_permutation(baseline)
    assert legacy_scalar_checksum(baseline) == legacy_scalar_checksum(permuted)
    legacy = legacy_checksum_agreement(baseline, permuted)
    assert legacy["within_tolerance"], (
        "the legacy checksum check is expected to accept the spatial "
        "permutation - that is the defect this fixture documents"
    )


def test_the_full_tensor_check_rejects_a_sign_flip():
    baseline = _rollout()
    flipped = _sign_flip(baseline)
    report = compare_rollout_outputs(baseline, flipped, reference=_reference(4, 6, 16))
    assert not report["passed"]
    assert "relative_l2" in report["failures"]
    assert report["relative_l2"] == 2.0
    assert report["max_abs_error"] > 0
    assert report["nrmse_delta"] is not None
    # every time step is reported, not only one scalar
    assert len(report["per_time_step"]) == baseline.shape[1]
    assert all(step["max_abs_error"] > 0 for step in report["per_time_step"])


def test_the_full_tensor_check_rejects_a_spatial_permutation():
    baseline = _rollout()
    permuted = _spatial_permutation(baseline)
    report = compare_rollout_outputs(baseline, permuted, reference=_reference(4, 6, 16))
    assert not report["passed"]
    assert "relative_l2" in report["failures"]
    assert report["relative_l2"] > 1.0
    assert report["max_abs_error"] > 0
    assert len(report["per_time_step"]) == baseline.shape[1]


def test_the_full_tensor_check_accepts_an_identical_output():
    baseline = _rollout()
    report = compare_rollout_outputs(baseline, baseline.copy(), reference=_reference(4, 6, 16))
    assert report["passed"]
    assert report["failures"] == []
    assert report["max_abs_error"] == 0.0
    assert report["relative_l2"] == 0.0


def test_the_full_tensor_check_accepts_a_perturbation_inside_the_tolerance():
    baseline = _rollout()
    rng = np.random.default_rng(7)
    perturbed = baseline + 1e-4 * rng.standard_normal(baseline.shape)
    report = compare_rollout_outputs(baseline, perturbed, reference=_reference(4, 6, 16))
    assert report["passed"], report["failures"]


def test_the_full_tensor_check_rejects_non_finite_values():
    baseline = _rollout()
    broken = baseline.copy()
    broken[0, 0, 0, 0] = np.nan
    report = compare_rollout_outputs(baseline, broken, reference=_reference(4, 6, 16))
    assert not report["passed"]
    assert "nonfinite" in report["failures"]
    assert report["nonfinite_candidate"] == 1


def test_the_tolerance_is_frozen_and_declared():
    """The tolerance is a frozen constant, not a per-candidate value."""
    assert ACCURACY_TOLERANCE.max_abs_error == 5.0e-2
    assert ACCURACY_TOLERANCE.max_relative_l2 == 1.0e-2
    assert ACCURACY_TOLERANCE.max_rollout_nrmse_delta == 1.0e-2
    assert ACCURACY_TOLERANCE.max_nonfinite_count == 0


def _print_evidence() -> int:
    baseline = _rollout()
    reference = _reference(4, 6, 16)
    cases = [
        ("identical", baseline.copy()),
        ("small perturbation (1e-4)", baseline + 1e-4),
        ("sign flip", _sign_flip(baseline)),
        ("spatial permutation", _spatial_permutation(baseline)),
    ]
    header = (
        f"{'case':28s} {'checksum(base)':>15s} {'checksum(cand)':>15s} "
        f"{'legacy verdict':>15s} {'max abs err':>13s} {'rel L2':>10s} "
        f"{'new verdict':>12s}"
    )
    print(header)
    print("-" * len(header))
    failures = 0
    for label, candidate in cases:
        legacy = legacy_checksum_agreement(baseline, candidate)
        report = compare_rollout_outputs(baseline, candidate, reference=reference)
        print(
            f"{label:28s} {legacy['baseline_checksum']:15.8f} "
            f"{legacy['candidate_checksum']:15.8f} "
            f"{('accepted' if legacy['within_tolerance'] else 'rejected'):>15s} "
            f"{report['max_abs_error']:13.3e} {report['relative_l2']:10.4f} "
            f"{('accepted' if report['passed'] else 'REJECTED'):>12s}"
        )
        if label in ("sign flip", "spatial permutation") and report["passed"]:
            failures += 1
    print()
    print(f"frozen tolerance: {ACCURACY_TOLERANCE.to_dict()}")
    print("the scalar checksum accepts the sign flip and the spatial")
    print("permutation; the element-wise check rejects both.")
    return 1 if failures else 0


def main() -> int:
    return _print_evidence()


if __name__ == "__main__":
    raise SystemExit(main())
