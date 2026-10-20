"""Training loop with optional physics regularization.

Identical optimizer family, stopping policy, validation schedule and
data pipeline are used for every compared condition. The only thing
that changes between the *data-only* and the *physics-regularized*
condition is the value of ``lambda_pde``; nothing else is re-tuned for
one condition alone.

Training-time cost is recorded separately from evaluation, because the
PDE term adds compute.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import project_root
from ..data.dataset import NormalizationStats, TrajectoryDataset
from ..models.base import TimeAdvanceModel


@dataclass
class TrainConfig:
    """Resolved training configuration."""

    epochs: int = 20
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_schedule: str = "cosine"
    warmup_steps: int = 100
    grad_clip: float = 1.0
    history: int = 1
    observation_dt: float = 0.01
    lambda_pde: float = 0.0
    lambda_ic: float = 0.0
    lambda_bc: float = 0.0
    pde_scale: float = 1.0
    data_scale: float = 1.0
    use_ic: bool = False
    use_bc: bool = False
    seed: int = 0
    device: str = "auto"
    log_every: int = 10
    eval_every: int = 1
    max_walltime_minutes: float = 45.0
    amp: bool = False
    ema_decay: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    run_id: str
    best_val_loss: float
    best_epoch: int
    history: list[dict[str, Any]]
    checkpoint_path: str
    walltime_seconds: float
    parameter_counts: dict[str, int]
    physics_grad_check: dict[str, Any] | None = None
    stopped_reason: str = "completed"
    determinism: dict[str, Any] = field(default_factory=dict)


class Trainer:
    """Train a one-step operator under a data or physics-regularized objective."""

    def __init__(
        self,
        model: TimeAdvanceModel,
        train_set: TrajectoryDataset,
        val_set: TrajectoryDataset,
        stats: NormalizationStats,
        config: TrainConfig,
        run_id: str,
        out_dir: str | Path,
        system: str,
        pde_kind: str,
        pde_params: dict[str, float] | None = None,
        param_vector: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.stats = stats
        self.config = config
        self.run_id = run_id
        self.out_dir = Path(out_dir)
        if not self.out_dir.is_absolute():
            self.out_dir = project_root() / self.out_dir
        self.system = system
        self.pde_kind = pde_kind
        self.pde_params = dict(pde_params or {})
        self.param_vector = param_vector
        self.device = self._resolve_device(config.device)
        self.history: list[dict[str, Any]] = []
        self.determinism: dict[str, Any] = {}
        # Bound the numerical-library thread count so that the pipeline
        # does not oversubscribe the shared CPU (see the hardware
        # profile: cpu_threads = 4, dataloader_workers = 0).
        thread_limit = int(config.extra.get("cpu_threads", 4))
        torch.set_num_threads(max(1, thread_limit))

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        if spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(spec)

    def _param_tensor(self, batch_size: int) -> torch.Tensor | None:
        if self.param_vector is None:
            return None
        return self.param_vector.to(self.device).unsqueeze(0).expand(batch_size, -1)

    # ------------------------------------------------------------------
    def _loss_for_batch(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        params: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        x = self.stats.normalize(history)
        y = self.stats.normalize(target)

        pred = self.model(x, params)
        data_loss = F.mse_loss(pred, y) / cfg.data_scale
        total = data_loss
        log = {"data": float(data_loss.detach())}

        if cfg.lambda_pde > 0:
            pde = self._physics_loss(pred, history, params)
            total = total + cfg.lambda_pde * pde
            log["pde"] = float(pde.detach())

        if cfg.use_ic and cfg.lambda_ic > 0:
            ic = F.mse_loss(pred[..., 0], x[..., 0])
            total = total + cfg.lambda_ic * ic / cfg.pde_scale
            log["ic"] = float(ic.detach())

        if cfg.use_bc and cfg.lambda_bc > 0:
            bc = self._boundary_loss(pred)
            total = total + cfg.lambda_bc * bc
            log["bc"] = float(bc.detach())

        log["total"] = float(total.detach())
        return total, log

    def _physics_loss(
        self,
        pred: torch.Tensor,
        history: torch.Tensor,
        params: torch.Tensor | None,
    ) -> torch.Tensor:
        """PDE residual loss in physical units on the *predicted* state.

        The residual is formed between the predicted state at t+dt and
        the (given) state at t, so the PDE term constrains the
        prediction rather than the ground truth. Gradients flow through
        the predicted state only.
        """
        from ..physics.losses import allen_cahn_physics_loss, burgers_physics_loss

        pred_phys = self.stats.denormalize(pred)
        # history: (B, H, *spatial); the last frame is the state at t.
        prev_phys = history[:, -1]
        if prev_phys.dim() == pred_phys.dim() - 1:
            prev_phys = prev_phys.unsqueeze(1)
        dt = self.config.observation_dt

        if self.pde_kind == "burgers":
            nu = self.pde_params.get("nu", 0.01)
            return burgers_physics_loss(pred_phys, prev_phys, dt=dt, nu=nu) / (
                self.config.pde_scale**2
            )
        if self.pde_kind == "allen_cahn":
            D = self.pde_params.get("D", 0.001)
            a = self.pde_params.get("a", 1.0)
            return allen_cahn_physics_loss(
                pred_phys, prev_phys, dt=dt, D=D, a=a
            ) / (self.config.pde_scale**2)
        raise ValueError(f"unknown pde kind {self.pde_kind}")

    def _boundary_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Periodic wrap mismatch (zero for the periodic benchmarks)."""
        if pred.dim() == 3:
            return F.mse_loss(pred[..., 0], pred[..., -1])
        return F.mse_loss(pred[..., 0, :], pred[..., -1, :])

    # ------------------------------------------------------------------
    def _loader(self, dataset: TrajectoryDataset, shuffle: bool) -> DataLoader:
        generator = torch.Generator().manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            generator=generator if shuffle else None,
        )

    def train(self) -> TrainResult:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        # Reproducibility: cuDNN autotuning selects different algorithms
        # from run to run, which makes a seeded run irreproducible (it
        # was measured here: the same 1D U-Net configuration re-run at
        # the same seed changed its validation loss by ~20%). Turning
        # autotuning off and selecting deterministic kernels makes a
        # seeded run reproducible.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.determinism = {
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "note": "GPU reductions and FFT kernels are not bitwise "
                    "deterministic; a seeded run is reproducible in "
                    "configuration and data order, not necessarily in the "
                    "last floating-point digits",
        }
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        loader = self._loader(self.train_set, shuffle=True)
        steps_per_epoch = max(len(loader), 1)
        total_steps = cfg.epochs * steps_per_epoch

        if cfg.lr_schedule == "cosine":
            def lr_lambda(step: int) -> float:
                if step < cfg.warmup_steps:
                    return (step + 1) / max(cfg.warmup_steps, 1)
                progress = (step - cfg.warmup_steps) / max(
                    total_steps - cfg.warmup_steps, 1
                )
                return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

        start = time.time()
        best_val = float("inf")
        best_epoch = -1
        stop_reason = "completed"
        global_step = 0
        physics_check: dict[str, Any] | None = None

        for epoch in range(cfg.epochs):
            self.model.train()
            epoch_log: dict[str, float] = {}
            for batch in loader:
                history = batch["history"].to(self.device)
                target = batch["target"].to(self.device)
                params = self._param_tensor(history.shape[0])

                if physics_check is None and cfg.lambda_pde > 0 and global_step == 0:
                    physics_check = self._check_physics_gradients(history, params)

                optimizer.zero_grad(set_to_none=True)
                loss, log = self._loss_for_batch(history, target, params)
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                )
                optimizer.step()
                scheduler.step()
                global_step += 1

                for k, v in log.items():
                    epoch_log[k] = epoch_log.get(k, 0.0) + v
                epoch_log["grad_norm"] = epoch_log.get("grad_norm", 0.0) + grad_norm
                epoch_log["lr"] = scheduler.get_last_lr()[0]

            n = max(len(loader), 1)
            for k in list(epoch_log):
                epoch_log[k] /= n

            val_loss, val_log = self._validate()
            record = {
                "epoch": epoch,
                "train": epoch_log,
                "val": val_log,
                "elapsed_seconds": round(time.time() - start, 2),
            }
            self.history.append(record)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                self._save_checkpoint("best.pt", epoch, optimizer, scheduler, global_step)

            if (time.time() - start) / 60.0 > cfg.max_walltime_minutes:
                stop_reason = "walltime_budget_exceeded"
                break

        self._save_checkpoint("last.pt", best_epoch, optimizer, scheduler, global_step)
        walltime = time.time() - start

        return TrainResult(
            run_id=self.run_id,
            best_val_loss=best_val,
            best_epoch=best_epoch,
            history=self.history,
            checkpoint_path=str(self.out_dir / "best.pt"),
            walltime_seconds=walltime,
            parameter_counts=self._param_counts(),
            physics_grad_check=physics_check,
            stopped_reason=stop_reason,
            determinism=self.determinism,
        )

    # ------------------------------------------------------------------
    def _check_physics_gradients(
        self, history: torch.Tensor, params: torch.Tensor | None
    ) -> dict[str, Any]:
        """Verify the PDE term contributes nonzero, finite gradients.

        The gradient of the physics loss alone is computed and inspected.
        A zero or non-finite gradient is a hard failure of the physics
        regularizer and is recorded rather than ignored.
        """
        x = self.stats.normalize(history)
        pred = self.model(x, params)
        pde = self._physics_loss(pred, history, params)
        self.model.zero_grad(set_to_none=True)
        pde.backward(retain_graph=True)
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        total = float(sum(g.abs().sum().item() for g in grads)) if grads else 0.0
        finite = all(bool(torch.isfinite(g).all()) for g in grads)
        nonzero = total > 0.0
        return {
            "pde_loss_value": float(pde.detach()),
            "grad_l1_sum": total,
            "grads_finite": finite,
            "grads_nonzero": nonzero,
            "n_grad_tensors": len(grads),
            "passed": bool(finite and nonzero),
        }

    @torch.no_grad()
    def _validate(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        loader = self._loader(self.val_set, shuffle=False)
        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            history = batch["history"].to(self.device)
            target = batch["target"].to(self.device)
            params = self._param_tensor(history.shape[0])
            loss, log = self._loss_for_batch(history, target, params)
            for k, v in log.items():
                totals[k] = totals.get(k, 0.0) + v
            count += 1
        count = max(count, 1)
        return totals.get("total", float("inf")) / count, {
            k: v / count for k, v in totals.items()
        }

    def _param_counts(self) -> dict[str, int]:
        from ..models.base import count_parameters

        return count_parameters(self.model)

    def _save_checkpoint(
        self, name: str, epoch: int, optimizer, scheduler, global_step: int
    ) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": asdict(self.config),
                "run_id": self.run_id,
                "system": self.system,
                "pde_kind": self.pde_kind,
                "pde_params": self.pde_params,
                "normalization": self.stats.to_dict(),
                "rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
            path,
        )
        return path


def load_checkpoint(path: str | Path, model: TimeAdvanceModel, device="cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt
