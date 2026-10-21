"""Training loop with optional physics regularization.

Identical optimizer family, stopping policy, validation schedule and
data pipeline are used for every compared condition. The only thing
that changes between the *data-only* and the *physics-regularized*
condition is the value of ``lambda_pde``; nothing else is re-tuned for
one condition alone.

Training-time cost is recorded separately from evaluation, because the
PDE term adds compute.

Resume
------
A checkpoint written by this module carries the *whole* training state:
model, optimizer, scheduler, the epoch the state belongs to, the batch
index inside that epoch, the global step, the shuffle generator state
and the epoch's data order, the Python/NumPy/CPU/CUDA RNG states, the
normalization constants, the configuration, the data identity and the
stopping policy.  Loading it and continuing is a genuine resume: an
interrupted run ends in the same state as an uninterrupted one.

``last.pt`` records the *current* position of the training loop (the
number of completed epochs and the batch index inside the next epoch),
never the epoch that happened to have the best validation loss.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..config import project_root
from ..data.dataset import NormalizationStats, TrajectoryDataset
from ..models.base import TimeAdvanceModel
from ..utils import seed_everything


class TrainingInterrupted(RuntimeError):
    """Raised when a run is stopped on purpose so that it can be resumed."""


class PhysicsGradientCheckError(RuntimeError):
    """The physics regulariser produced no finite, non-zero gradient.

    This is a hard failure: the run stops instead of continuing with a
    recorded ``passed=False``.
    """


class ResumeMismatchError(RuntimeError):
    """The checkpoint does not belong to this run's configuration or data."""


#: Version of the checkpoint/resume schema.
RESUME_SCHEMA_VERSION = 2

#: Configuration fields that may change between an interruption and the
#: resume: they define the *stopping policy*, not the training state.
RESUME_CONFIG_EXEMPT = ("epochs", "max_walltime_minutes")


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
    epochs_completed: int = 0
    last_epoch: int = -1
    global_step: int = 0
    resumed_from: str | None = None


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
        data_identity: dict[str, Any] | None = None,
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
        self.data_identity = dict(data_identity or {})
        self.device = self._resolve_device(config.device)
        self.history: list[dict[str, Any]] = []
        self.determinism: dict[str, Any] = {}
        # The shuffle order is driven by an explicit generator whose state
        # is part of the checkpoint, so that a resumed run sees exactly
        # the same data order as an uninterrupted one.
        self._shuffle_generator = torch.Generator()
        self._shuffle_generator.manual_seed(int(config.seed))
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
    def _epoch_batches(self) -> list[list[int]]:
        """The batch index lists of one epoch, in shuffled order.

        The permutation is drawn from the checkpointed shuffle
        generator, so the order is a deterministic function of the
        generator state and survives a resume.
        """
        n = len(self.train_set)
        perm = torch.randperm(n, generator=self._shuffle_generator).tolist()
        bs = max(1, int(self.config.batch_size))
        return [perm[i : i + bs] for i in range(0, n, bs)]

    def _val_batches(self) -> list[list[int]]:
        """The batch index lists of the validation pass (never shuffled)."""
        n = len(self.val_set)
        bs = max(1, int(self.config.batch_size))
        return [list(range(i, min(i + bs, n))) for i in range(0, n, bs)]

    def _batches(self, dataset: TrajectoryDataset, batches) -> Any:
        """Yield the stacked tensors of the given index lists.

        The batches are assembled here rather than by a ``DataLoader``
        because a ``DataLoader`` iterator draws a *base seed* from the
        global RNG when it is created.  That draw would be repeated
        after a resume and would make the two runs differ in their RNG
        state, which is exactly what the resume has to reproduce.
        """
        for indices in batches:
            items = [dataset[i] for i in indices]
            yield {
                "history": torch.stack([it["history"] for it in items]),
                "target": torch.stack([it["target"] for it in items]),
            }

    def _stopping_policy(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "max_epochs": int(cfg.epochs),
            "max_walltime_minutes": float(cfg.max_walltime_minutes),
            "validation_every": int(cfg.eval_every),
            "physics_gradient_gate": "hard" if cfg.lambda_pde > 0 else "not_applicable",
            "walltime_stop_is_graceful": True,
        }

    def _config_payload(self) -> dict[str, Any]:
        return asdict(self.config)

    # ------------------------------------------------------------------
    def train(
        self,
        resume_from: str | Path | None = None,
        interrupt_after_batches: int | None = None,
    ) -> TrainResult:
        """Train the model.

        Parameters
        ----------
        resume_from:
            A checkpoint written by this module.  The whole training
            state is restored and the loop continues where it stopped.
        interrupt_after_batches:
            Stop the run after this many optimizer steps *in this call*
            and write ``last.pt``.  Used to exercise the resume path;
            the run ends with :class:`TrainingInterrupted`.
        """
        cfg = self.config
        # Re-seed with the same value the experiment runner used before
        # constructing the model, so the data order and any stochastic
        # layer are deterministic as well.
        seed_everything(cfg.seed)
        self.determinism = {
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "seeded_before_model_construction": True,
            "note": "GPU reductions and FFT kernels are not bitwise "
                    "deterministic; a seeded run is reproducible in "
                    "configuration, initialisation and data order, but not "
                    "necessarily in the last floating-point digits",
        }
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        steps_per_epoch = max(1, len(self._epoch_batches_preview()))
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
        epoch = 0
        epochs_completed = 0
        batch_index = 0
        epoch_batches: list[list[int]] | None = None
        epoch_log: dict[str, float] = {}
        epoch_lr = 0.0
        resumed_from: str | None = None

        if resume_from is not None:
            state = self._load_resume_state(resume_from)
            self.model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["rng_state"])
            np.random.set_state(state["numpy_rng_state"])
            random.setstate(state["python_rng_state"])
            if state.get("cuda_rng_state") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda_rng_state"])
            self._shuffle_generator.set_state(state["shuffle_generator_state"])
            self.history = list(state.get("history") or [])
            best_val = float(state.get("best_val_loss", float("inf")))
            best_epoch = int(state.get("best_epoch", -1))
            global_step = int(state.get("global_step", 0))
            epoch = int(state.get("epoch", 0))
            epochs_completed = int(state.get("epochs_completed", epoch))
            batch_index = int(state.get("batch_index", 0))
            epoch_batches = state.get("epoch_batches")
            # The accumulator of the interrupted epoch is part of the
            # state: without it the resumed epoch would report the mean
            # over the batches it happened to run.
            epoch_log = dict(state.get("epoch_log") or {})
            epoch_lr = float((state.get("epoch_log") or {}).get("lr", 0.0))
            physics_check = state.get("physics_grad_check")
            resumed_from = str(resume_from)

        steps_this_call = 0
        while epoch < cfg.epochs:
            if epoch_batches is None:
                epoch_batches = self._epoch_batches()
                # A fresh epoch starts with a fresh accumulator.
                epoch_log = {}
            loader = self._batches(self.train_set, epoch_batches)
            self.model.train()
            n_batches = len(epoch_batches)
            for b_idx, batch in enumerate(loader):
                if b_idx < batch_index:
                    continue
                history = batch["history"].to(self.device)
                target = batch["target"].to(self.device)
                params = self._param_tensor(history.shape[0])

                if physics_check is None and cfg.lambda_pde > 0 and global_step == 0:
                    physics_check = self._check_physics_gradients(history, params)
                    self.model.zero_grad(set_to_none=True)
                    if not physics_check["passed"]:
                        self._save_checkpoint(
                            "last.pt",
                            epoch=epoch,
                            epochs_completed=epochs_completed,
                            batch_index=b_idx,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            global_step=global_step,
                            epoch_batches=epoch_batches,
                            epoch_log=epoch_log,
                            best_val=best_val,
                            best_epoch=best_epoch,
                            physics_check=physics_check,
                        )
                        error = PhysicsGradientCheckError(
                            "the physics term produced no usable gradient "
                            f"(grads_finite={physics_check['grads_finite']}, "
                            f"grads_nonzero={physics_check['grads_nonzero']}, "
                            f"grad_l1_sum={physics_check['grad_l1_sum']:.6g}); "
                            "the run is stopped"
                        )
                        error.physics_grad_check = physics_check
                        raise error

                optimizer.zero_grad(set_to_none=True)
                loss, log = self._loss_for_batch(history, target, params)
                loss.backward()
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                )
                optimizer.step()
                scheduler.step()
                global_step += 1
                steps_this_call += 1

                for k, v in log.items():
                    epoch_log[k] = epoch_log.get(k, 0.0) + v
                epoch_log["grad_norm"] = epoch_log.get("grad_norm", 0.0) + grad_norm
                # The learning rate is a state, not an accumulator: it is
                # reported as the rate of the last step of the epoch.
                epoch_lr = float(scheduler.get_last_lr()[0])

                if interrupt_after_batches is not None and steps_this_call >= interrupt_after_batches:
                    self._save_checkpoint(
                        "last.pt",
                        epoch=epoch,
                        epochs_completed=epochs_completed,
                        batch_index=b_idx + 1,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch_batches=epoch_batches,
                        epoch_log=epoch_log,
                        best_val=best_val,
                        best_epoch=best_epoch,
                        physics_check=physics_check,
                    )
                    raise TrainingInterrupted(
                        f"interrupted after {steps_this_call} steps in epoch {epoch}"
                    )

            for k in list(epoch_log):
                epoch_log[k] /= max(n_batches, 1)
            epoch_log["lr"] = epoch_lr

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
                self._save_checkpoint(
                    "best.pt",
                    epoch=epoch + 1,
                    epochs_completed=epoch + 1,
                    batch_index=0,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch_batches=None,
                    epoch_log=None,
                    best_val=best_val,
                    best_epoch=best_epoch,
                    physics_check=physics_check,
                )

            epochs_completed = epoch + 1
            epoch = epochs_completed
            batch_index = 0
            epoch_batches = None
            self._save_checkpoint(
                "last.pt",
                epoch=epoch,
                epochs_completed=epochs_completed,
                batch_index=0,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                epoch_batches=None,
                epoch_log=None,
                best_val=best_val,
                best_epoch=best_epoch,
                physics_check=physics_check,
            )

            if (time.time() - start) / 60.0 > cfg.max_walltime_minutes:
                stop_reason = "walltime_budget_exceeded"
                break

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
            epochs_completed=epochs_completed,
            last_epoch=epochs_completed - 1,
            global_step=global_step,
            resumed_from=resumed_from,
        )

    def _epoch_batches_preview(self) -> list[list[int]]:
        """The number of batches per epoch, without touching the generator."""
        n = len(self.train_set)
        bs = max(1, int(self.config.batch_size))
        return [list(range(i, min(i + bs, n))) for i in range(0, n, bs)]

    # ------------------------------------------------------------------
    def _check_physics_gradients(
        self, history: torch.Tensor, params: torch.Tensor | None
    ) -> dict[str, Any]:
        """Verify the PDE term contributes nonzero, finite gradients.

        The gradient of the physics loss alone is computed and inspected.
        A zero or non-finite gradient is a hard failure of the physics
        regularizer: the caller stops the run.
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
        loader = self._batches(self.val_set, self._val_batches())
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

    # ------------------------------------------------------------------
    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        epochs_completed: int,
        batch_index: int,
        optimizer,
        scheduler,
        global_step: int,
        epoch_batches: list[list[int]] | None,
        epoch_log: dict[str, float] | None,
        best_val: float,
        best_epoch: int,
        physics_check: dict[str, Any] | None,
    ) -> dict[str, Any]:
        from ..evaluation.metrics import canonical_json_sha256, normalization_identity

        config_payload = self._config_payload()
        return {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            # The *current* position of the training loop, never the best
            # epoch: "epoch" is the index of the next epoch to run (i.e.
            # the number of completed epochs) and "batch_index" says how
            # far into that epoch the run got.
            "epoch": int(epoch),
            "epochs_completed": int(epochs_completed),
            "last_executed_epoch": int(epochs_completed - 1),
            "batch_index": int(batch_index),
            "global_step": int(global_step),
            "epoch_batches": epoch_batches,
            "epoch_log": dict(epoch_log) if epoch_log else None,
            "shuffle_generator_state": self._shuffle_generator.get_state(),
            "data_position": {
                "epoch": int(epoch),
                "batch_index": int(batch_index),
                "samples_consumed": int(global_step) * int(self.config.batch_size),
            },
            "config": config_payload,
            "config_sha256": canonical_json_sha256(config_payload),
            "data_identity": self.data_identity,
            "data_identity_sha256": canonical_json_sha256(self.data_identity),
            "stopping_policy": self._stopping_policy(),
            "normalization": self.stats.to_dict(),
            "normalization_sha256": normalization_identity(self.stats)["sha256"],
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "history": list(self.history),
            "physics_grad_check": physics_check,
            "run_id": self.run_id,
            "system": self.system,
            "pde_kind": self.pde_kind,
            "pde_params": self.pde_params,
        }

    def _save_checkpoint(
        self,
        name: str,
        *,
        epoch: int,
        epochs_completed: int,
        batch_index: int,
        optimizer,
        scheduler,
        global_step: int,
        epoch_batches: list[list[int]] | None,
        epoch_log: dict[str, float] | None,
        best_val: float,
        best_epoch: int,
        physics_check: dict[str, Any] | None,
    ) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / name
        payload = self._checkpoint_payload(
            epoch=epoch,
            epochs_completed=epochs_completed,
            batch_index=batch_index,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch_batches=epoch_batches,
            epoch_log=epoch_log,
            best_val=best_val,
            best_epoch=best_epoch,
            physics_check=physics_check,
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def _load_resume_state(self, path: str | Path) -> dict[str, Any]:
        """Load and *validate* a checkpoint for continuation."""
        from ..evaluation.metrics import canonical_json_sha256

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        version = int(ckpt.get("resume_schema_version", 0))
        if version < RESUME_SCHEMA_VERSION:
            raise ResumeMismatchError(
                f"checkpoint {path} uses resume schema {version}; "
                f"this build requires {RESUME_SCHEMA_VERSION}"
            )

        config_payload = self._config_payload()
        stored_config = ckpt.get("config") or {}
        differing = [
            key
            for key in set(config_payload) | set(stored_config)
            if key not in RESUME_CONFIG_EXEMPT
            and config_payload.get(key) != stored_config.get(key)
        ]
        if differing:
            raise ResumeMismatchError(
                "the checkpoint was written with a different configuration: "
                + ", ".join(sorted(differing))
            )

        stored_data = ckpt.get("data_identity") or {}
        if canonical_json_sha256(stored_data) != canonical_json_sha256(self.data_identity):
            raise ResumeMismatchError(
                "the checkpoint was written for a different dataset; "
                "resuming it would continue on the wrong data"
            )

        stored_norm = ckpt.get("normalization") or {}
        if canonical_json_sha256(stored_norm) != canonical_json_sha256(
            self.stats.to_dict()
        ):
            raise ResumeMismatchError(
                "the checkpoint was written with different normalisation constants"
            )

        missing = [
            key
            for key in (
                "model",
                "optimizer",
                "scheduler",
                "epoch",
                "global_step",
                "shuffle_generator_state",
                "rng_state",
                "numpy_rng_state",
                "python_rng_state",
                "normalization",
                "config",
                "data_identity",
                "stopping_policy",
            )
            if key not in ckpt
        ]
        if missing:
            raise ResumeMismatchError(
                "the checkpoint is missing the state required for a resume: "
                + ", ".join(missing)
            )
        return ckpt


def load_checkpoint(path: str | Path, model: TimeAdvanceModel, device="cpu") -> dict:
    """Load the weights of a checkpoint into a model.

    This is *not* a resume: it restores the parameters only.  Use
    :meth:`Trainer.train` with ``resume_from`` for a continuation.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt
