"""Experiment orchestration.

A run is fully described by its manifest: date, Git revision and dirty
status, config hash, dataset/split hashes, seeds, parameter counts,
package versions, hardware, wall time, peak RAM/VRAM, checkpoint
selection, failures and the exact continuation command.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import config_hash, load_yaml, project_root
from ..data.dataset import (
    NormalizationStats,
    TrajectoryDataset,
    compute_normalization_stats,
    load_manifest,
    split_paths,
)
from ..data.generate import environment_snapshot, git_revision
from ..evaluation.metrics import (
    nrmse_per_snapshot,
    nrmse_per_trajectory,
    summarize_nrmse,
)
from ..evaluation.rollout import rollout
from ..models.base import count_parameters
from ..models.registry import build_model_from_config
from ..resources.preflight import HardwareProfile, preflight
from ..training.trainer import TrainConfig, Trainer, load_checkpoint


@dataclass
class ExperimentConfig:
    """Resolved experiment configuration."""

    name: str
    system: str
    dataset_dir: str
    architecture: str
    condition: str
    seed: int
    epochs: int
    batch_size: int
    lr: float
    history: int = 1
    observation_dt: float = 0.01
    lambda_pde: float = 0.0
    lambda_ic: float = 0.0
    lambda_bc: float = 0.0
    pde_scale: float = 1.0
    data_scale: float = 1.0
    use_ic: bool = False
    use_bc: bool = False
    weight_decay: float = 0.0
    lr_schedule: str = "cosine"
    warmup_steps: int = 100
    grad_clip: float = 1.0
    device: str = "auto"
    max_walltime_minutes: float = 45.0
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> tuple[ExperimentConfig, dict]:
    """Load and flatten an experiment YAML."""
    raw = load_yaml(path)
    model_cfg = raw.get("model", {}) or {}
    train_cfg = raw.get("train", {}) or {}
    data_cfg = raw.get("data", {}) or {}

    flat: dict[str, Any] = {
        "name": raw.get("name", "experiment"),
        "system": raw["system"],
        "dataset_dir": data_cfg.get("dataset_dir", f"data/{raw['system']}"),
        "architecture": model_cfg.get("architecture", "fno1d"),
        "condition": raw.get("condition", "data_only"),
        "seed": int(raw.get("seed", 0)),
        "epochs": int(train_cfg.get("epochs", 20)),
        "batch_size": int(train_cfg.get("batch_size", 8)),
        "lr": float(train_cfg.get("lr", 1e-3)),
        "history": int(model_cfg.get("history", 1)),
        "observation_dt": float(raw.get("observation_dt", 0.01)),
        "lambda_pde": float(raw.get("lambda_pde", 0.0)),
        "lambda_ic": float(raw.get("lambda_ic", 0.0)),
        "lambda_bc": float(raw.get("lambda_bc", 0.0)),
        "pde_scale": float(raw.get("pde_scale", 1.0)),
        "data_scale": float(raw.get("data_scale", 1.0)),
        "use_ic": bool(raw.get("use_ic", False)),
        "use_bc": bool(raw.get("use_bc", False)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "lr_schedule": train_cfg.get("lr_schedule", "cosine"),
        "warmup_steps": int(train_cfg.get("warmup_steps", 100)),
        "grad_clip": float(train_cfg.get("grad_clip", 1.0)),
        "device": train_cfg.get("device", "auto"),
        "max_walltime_minutes": float(
            train_cfg.get("max_walltime_minutes", 45.0)
        ),
        "model_kwargs": {
            k: v for k, v in model_cfg.items() if k not in ("architecture", "history")
        },
        "tags": list(raw.get("tags", []) or []),
    }
    return ExperimentConfig(**flat), raw


def run_id_for(cfg: ExperimentConfig) -> str:
    return f"{cfg.system}_{cfg.architecture}_{cfg.condition}_seed{cfg.seed}"


class ExperimentRunner:
    """Execute one training condition end to end and write its manifest."""

    def __init__(self, cfg: ExperimentConfig, raw_config: dict, profile: HardwareProfile):
        self.cfg = cfg
        self.raw_config = raw_config
        self.profile = profile
        self.run_id = run_id_for(cfg)
        self.run_dir = project_root() / "runs" / self.run_id

    # ------------------------------------------------------------------
    def _dataset_manifest(self) -> dict[str, Any]:
        paths = split_paths(self.cfg.dataset_dir, self.cfg.system)
        manifest_path = Path(self.cfg.dataset_dir) / f"{self.cfg.system}_manifest.json"
        manifest_path = (
            manifest_path if manifest_path.is_absolute()
            else project_root() / manifest_path
        )
        if manifest_path.exists():
            return load_manifest(manifest_path)
        return {"error": "dataset manifest not found", "path": str(manifest_path)}

    def _param_vector(self, manifest: dict) -> torch.Tensor | None:
        """PDE parameter channels supplied to the model, in physical units."""
        n_params = int(self.cfg.model_kwargs.get("n_params", 0) or 0)
        if n_params == 0:
            return None
        params = (manifest.get("system", {}) or {}).get("params", {}) or {}
        keys = sorted(params)
        if len(keys) != n_params:
            raise ValueError(
                f"model expects {n_params} parameter channels but system declares {len(keys)}"
            )
        return torch.tensor([float(params[k]) for k in keys], dtype=torch.float32)

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        pre = preflight(self.profile, require_gpu=(self.cfg.device != "cpu"))

        manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "created_utc": started_utc,
            "status": "running",
            "experiment": self.cfg.to_dict(),
            "raw_config": self.raw_config,
            "config_hash": config_hash(self.raw_config),
            "dataset_manifest": self._dataset_manifest(),
            "git": git_revision(project_root()),
            "environment": environment_snapshot(),
            "preflight": pre,
            "continuation_command": (
                f"pdelab train --config {self.cfg.name}.yaml --seed {self.cfg.seed}"
            ),
        }

        if not pre["ok"]:
            manifest["status"] = "blocked"
            manifest["block_reason"] = "; ".join(pre["violations"])
            manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._write_manifest(manifest)
            return manifest

        try:
            paths = split_paths(self.cfg.dataset_dir, self.cfg.system)
            stats = compute_normalization_stats(paths["train"])
            manifest["normalization"] = stats.to_dict()

            train_set = TrajectoryDataset(
                paths["train"], stats, history=self.cfg.history, stride=1
            )
            val_set = TrajectoryDataset(
                paths["val"], stats, history=self.cfg.history, stride=1
            )

            model = build_model_from_config(
                {
                    "architecture": self.cfg.architecture,
                    "history": self.cfg.history,
                    **self.cfg.model_kwargs,
                },
                dimension=self._dimension(),
            )
            manifest["parameter_counts"] = count_parameters(model)

            train_cfg = TrainConfig(
                epochs=self.cfg.epochs,
                batch_size=self.cfg.batch_size,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                lr_schedule=self.cfg.lr_schedule,
                warmup_steps=self.cfg.warmup_steps,
                grad_clip=self.cfg.grad_clip,
                history=self.cfg.history,
                observation_dt=self.cfg.observation_dt,
                lambda_pde=self.cfg.lambda_pde,
                lambda_ic=self.cfg.lambda_ic,
                lambda_bc=self.cfg.lambda_bc,
                pde_scale=self.cfg.pde_scale,
                data_scale=self.cfg.data_scale,
                use_ic=self.cfg.use_ic,
                use_bc=self.cfg.use_bc,
                seed=self.cfg.seed,
                device=self.cfg.device,
                max_walltime_minutes=self.cfg.max_walltime_minutes,
            )

            trainer = Trainer(
                model=model,
                train_set=train_set,
                val_set=val_set,
                stats=stats,
                config=train_cfg,
                run_id=self.run_id,
                out_dir=self.run_dir,
                system=self.cfg.system,
                pde_kind=self.cfg.system,
                pde_params=self._pde_params(manifest),
                param_vector=self._param_vector(manifest["dataset_manifest"]),
            )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            result = trainer.train()

            manifest["training"] = {
                "best_val_loss": result.best_val_loss,
                "best_epoch": result.best_epoch,
                "walltime_seconds": result.walltime_seconds,
                "stopped_reason": result.stopped_reason,
                "physics_grad_check": result.physics_grad_check,
                "history": result.history,
                "checkpoint_path": result.checkpoint_path,
            }
            manifest["peak_cuda_allocated_gib"] = (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            )
            manifest["peak_host_rss_gib"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            )
            manifest["status"] = "completed"
        except Exception as exc:  # noqa: BLE001 - failures must be recorded
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            manifest["traceback"] = traceback.format_exc()
        finally:
            manifest["walltime_seconds"] = round(time.time() - started, 3)
            manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._write_manifest(manifest)
        return manifest

    # ------------------------------------------------------------------
    def _dimension(self) -> int:
        return 2 if self.cfg.system == "allen_cahn" else 1

    def _pde_params(self, manifest: dict) -> dict[str, float]:
        params = (manifest.get("dataset_manifest", {}).get("system", {}) or {}).get(
            "params", {}
        )
        return {k: float(v) for k, v in (params or {}).items()}

    def _write_manifest(self, manifest: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "manifest.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)
