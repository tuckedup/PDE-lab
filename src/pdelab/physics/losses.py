"""Composite physics-regularized training objective.

    total_loss = data_loss + lambda_pde * pde_loss
                 + lambda_ic  * ic_loss
                 + lambda_bc  * bc_loss

Only terms that are *applicable* to the chosen interface are included:

* ``lambda_ic`` is used only when the model does **not** receive the
  initial condition exactly. In this project the initial snapshot is
  always supplied exactly as an input channel, so a trainable IC penalty
  would be redundant; it is therefore disabled by default and any use is
  recorded explicitly in the run manifest.
* ``lambda_bc`` is used only for non-periodic boundaries. Both benchmarks
  are periodic, and the boundary condition is enforced *structurally* by
  the spectral operators and by circular padding in the U-Net. That is
  documented as a structural enforcement, not reported as the outcome of
  an artificial boundary penalty.

Residuals are evaluated in **physical units**: the model output is
denormalized before the residual is formed, and the term scales below
are frozen per experiment configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .operators import allen_cahn_residual, burgers_residual


@dataclass
class PhysicsLossConfig:
    """Weights and scales for the composite objective."""

    lambda_pde: float = 0.0
    lambda_ic: float = 0.0
    lambda_bc: float = 0.0
    # Frozen scales applied to each loss term so that units do not
    # dominate one another. Chosen from the *training* set magnitudes.
    data_scale: float = 1.0
    pde_scale: float = 1.0
    ic_scale: float = 1.0
    bc_scale: float = 1.0
    use_ic: bool = False
    use_bc: bool = False


@dataclass
class LossTerms:
    """Individual loss components, kept separate for logging."""

    total: torch.Tensor
    data: torch.Tensor
    pde: torch.Tensor
    ic: torch.Tensor | None = None
    bc: torch.Tensor | None = None
    extras: dict = field(default_factory=dict)

    def as_log(self) -> dict[str, float]:
        log = {
            "loss/total": float(self.total.detach()),
            "loss/data": float(self.data.detach()),
            "loss/pde": float(self.pde.detach()),
        }
        if self.ic is not None:
            log["loss/ic"] = float(self.ic.detach())
        if self.bc is not None:
            log["loss/bc"] = float(self.bc.detach())
        for key, value in self.extras.items():
            log[f"loss/{key}"] = float(value)
        return log


def burgers_physics_loss(
    u_next_phys: torch.Tensor,
    u_prev_phys: torch.Tensor,
    dt: float,
    nu: float,
) -> torch.Tensor:
    """Mean-squared Burgers residual in physical units."""
    r = burgers_residual(u_next_phys, u_prev_phys, dt=dt, nu=nu)
    return torch.mean(r**2)


def allen_cahn_physics_loss(
    u_next_phys: torch.Tensor,
    u_prev_phys: torch.Tensor,
    dt: float,
    D: float,
    a: float,
) -> torch.Tensor:
    """Mean-squared Allen-Cahn residual in physical units."""
    r = allen_cahn_residual(u_next_phys, u_prev_phys, dt=dt, D=D, a=a)
    return torch.mean(r**2)
