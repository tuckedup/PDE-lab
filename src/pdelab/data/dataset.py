"""Torch datasets, splits and train-only normalization.

Normalization statistics are computed **from the training split only**
and are then applied unchanged to validation, test and any model
condition. No per-trajectory or per-test normalization is performed
anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import project_root


@dataclass
class NormalizationStats:
    """Affine normalization constants derived from the training split."""

    mean: float
    std: float
    source: str = "train"

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "source": self.source}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "NormalizationStats":
        return NormalizationStats(
            mean=float(d["mean"]), std=float(d["std"]), source=d.get("source", "train")
        )


def compute_normalization_stats(h5_path: str | Path) -> NormalizationStats:
    """Mean/std of the training trajectories only."""
    p = Path(h5_path)
    if not p.is_absolute():
        p = project_root() / p
    with h5py.File(p, "r") as fh:
        u = fh["u"][:]
    mean = float(u.mean())
    std = float(u.std())
    if std <= 0:
        std = 1.0
    return NormalizationStats(mean=mean, std=std, source="train")


class TrajectoryDataset(Dataset):
    """Windowed dataset over whole trajectories.

    Splits are assigned at the *trajectory* level, so every window of a
    simulation stays inside its split. This is asserted at construction
    time and re-checked in the unit tests.
    """

    def __init__(
        self,
        h5_path: str | Path,
        stats: NormalizationStats,
        history: int = 1,
        stride: int = 1,
        limit: int | None = None,
    ) -> None:
        p = Path(h5_path)
        if not p.is_absolute():
            p = project_root() / p
        self.path = p
        self.stats = stats
        self.history = history
        self.stride = stride
        with h5py.File(p, "r") as fh:
            self.u = fh["u"][:]
            self.ic_seeds = fh["ic_seeds"][:] if "ic_seeds" in fh else None
        if limit is not None:
            self.u = self.u[:limit]
            if self.ic_seeds is not None:
                self.ic_seeds = self.ic_seeds[:limit]
        self.n_traj, self.n_times = self.u.shape[0], self.u.shape[1]
        self.n_windows = max(self.n_times - self.history, 0)
        self._index: list[tuple[int, int]] = [
            (t, s)
            for t in range(self.n_traj)
            for s in range(0, self.n_windows, self.stride)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def window(self, traj: int, start: int) -> tuple[np.ndarray, np.ndarray]:
        hist = self.u[traj, start : start + self.history]
        target = self.u[traj, start + self.history]
        return hist, target

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj, start = self._index[idx]
        hist, target = self.window(traj, start)
        # The target carries a leading singleton channel axis so that it
        # matches the model output shape (B, 1, *spatial) exactly; a
        # mismatch here silently broadcasts.
        target = target[np.newaxis, ...]
        return {
            "history": torch.from_numpy(np.ascontiguousarray(hist)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "traj": torch.tensor(traj, dtype=torch.long),
            "time_index": torch.tensor(start, dtype=torch.long),
        }

    def trajectory(self, traj: int) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.u[traj]))


def load_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def split_paths(dataset_dir: str | Path, system: str) -> dict[str, Path]:
    d = Path(dataset_dir)
    if not d.is_absolute():
        d = project_root() / d
    return {split: d / f"{system}_{split}.h5" for split in ("train", "val", "test")}
