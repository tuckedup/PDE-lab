"""Metric definitions.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

EPSILON = 1e-12
NEAR_ZERO_REF_THRESHOLD = 1e-8


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
    """Aggregate summary of a data-error metric."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    n_trajectories: int
    near_zero_reference_count: int
    epsilon: float = EPSILON

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "n_trajectories": self.n_trajectories,
            "near_zero_reference_count": self.near_zero_reference_count,
            "epsilon": self.epsilon,
        }


def summarize_nrmse(
    per_traj: np.ndarray,
    reference_energy: np.ndarray | None = None,
    epsilon: float = EPSILON,
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
    )


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
    """Copy the initial state forward for ``n_steps`` future snapshots."""
    initial = np.asarray(initial)
    return np.repeat(initial[None, ...], n_steps, axis=0)


def mean_conservation_error(
    trajectory: np.ndarray, reference_mean: float
) -> float:
    """Max absolute deviation of the spatial mean from the initial mean."""
    traj = np.asarray(trajectory, dtype=np.float64)
    axis = tuple(range(1, traj.ndim))
    means = traj.mean(axis=axis)
    return float(np.max(np.abs(means - reference_mean)))
