"""Autoregressive rollout evaluation.

A one-step model is rolled out over a horizon by feeding its own
prediction back as the next input. Two protocols are reported
separately and never conflated:

* **teacher-forced one-step**: the model is given the ground-truth
  previous state and predicts one step ahead.
* **free rollout**: the model consumes its own predictions for the whole
  horizon. This is the protocol used for the primary held-out metric.

History window
--------------
A model with ``history = H`` expects ``H`` past snapshots as input.
The rollout therefore keeps a *sliding window* of ``H`` frames: after
each step the oldest frame is dropped and the model's own prediction is
appended. Replacing the whole window by a single predicted frame is
only correct for ``H == 1`` and silently breaks every ``H > 1``
condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..data.dataset import NormalizationStats
from ..models.base import TimeAdvanceModel


@dataclass
class RolloutResult:
    """Predictions and per-step diagnostics for one split."""

    predictions: np.ndarray  # (n_traj, n_steps, ...)
    reference: np.ndarray  # (n_traj, n_steps, ...)
    step_errors: np.ndarray  # (n_traj, n_steps)

    def to_summary(self, epsilon: float = 1e-12) -> dict[str, Any]:
        from .metrics import nrmse_per_trajectory, summarize_nrmse

        per_traj = nrmse_per_trajectory(self.predictions, self.reference, epsilon)
        energy = np.sum(self.reference**2, axis=(1, 2)) if self.reference.ndim == 3 else None
        summary = summarize_nrmse(per_traj, energy, epsilon)
        return {
            "nrmse": summary.to_dict(),
            "per_trajectory": per_traj.tolist(),
            "step_error_mean": self.step_errors.mean(axis=0).tolist(),
        }


@torch.no_grad()
def rollout(
    model: TimeAdvanceModel,
    history: torch.Tensor,
    n_steps: int,
    stats: NormalizationStats,
    params: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Free-running rollout in **physical** units.

    Parameters
    ----------
    history:
        Initial history, shape ``(B, H, *spatial)``, physical units.
    n_steps:
        Number of future snapshots to predict.

    Notes
    -----
    The model consumes a *window* of ``H`` frames.  After every step the
    window is shifted by one frame and the model's own prediction is
    appended, so the last frame of the window is always the most recent
    prediction.  For ``H == 1`` this is identical to feeding the
    prediction straight back in.
    """
    model.eval()
    h = int(getattr(model, "history", history.shape[1]) or history.shape[1])
    h = max(1, min(h, history.shape[1]))
    x = stats.normalize(history.to(device))
    # One frame per history slot; the window is rebuilt every step.
    window = [x[:, i] for i in range(h)]
    outputs = []
    for _ in range(n_steps):
        # Marks a boundary for CUDA-graph capture (torch.compile
        # mode="reduce-overhead"). It is a no-op when CUDA graphs are
        # not in use, so the same rollout code path serves every
        # benchmark variant and the workload is identical.
        mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        if mark is not None:
            try:
                mark()
            except Exception:  # pragma: no cover - depends on torch build
                pass
        inp = torch.stack(window[-h:], dim=1)
        pred = model(inp, params)
        outputs.append(stats.denormalize(pred))
        # Feed the model's own normalised prediction back in as the most
        # recent frame of the window.
        window.append(pred[:, 0])
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def teacher_forced_one_step(
    model: TimeAdvanceModel,
    dataset,
    stats: NormalizationStats,
    params: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-step predictions with ground-truth inputs.

    Returns ``(predictions, references, trajectory_ids)``.  The values
    are **per window**: one value per dataset window, not per
    trajectory.  ``trajectory_ids`` says which trajectory each window
    belongs to, so that a caller can aggregate the windows into
    per-trajectory values instead of reporting a per-window mean under a
    per-trajectory label.
    """
    model.eval()
    preds, refs, traj_ids = [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        hist = item["history"].unsqueeze(0).to(device)
        target = item["target"].unsqueeze(0).to(device)
        pred = model(stats.normalize(hist), params)
        preds.append(stats.denormalize(pred).squeeze(0).cpu().numpy())
        refs.append(target.squeeze(0).cpu().numpy())
        traj_ids.append(int(item["traj"]))
    return np.stack(preds), np.stack(refs), np.asarray(traj_ids, dtype=np.int64)
