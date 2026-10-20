"""Unified model interface shared by all learned operators in PDELab.

Every learned condition implements a one-step time-advance map

    u(t) -> u(t + dt)

with an identical calling convention so that conditions can be compared
without interface differences confounding the result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TimeAdvanceModel(nn.Module, ABC):
    """Base class for one-step time-advance models.

    Parameters
    ----------
    history:
        Number of past snapshots supplied as input channels.
    n_params:
        Number of PDE parameter channels supplied to the model.
    """

    def __init__(self, history: int = 1, n_params: int = 0) -> None:
        super().__init__()
        self.history = history
        self.n_params = n_params

    @abstractmethod
    def forward(
        self,
        u_hist: torch.Tensor,
        params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance one step.

        Parameters
        ----------
        u_hist:
            Tensor of shape ``(B, history, *spatial)`` in *normalized* units.
        params:
            Optional tensor of shape ``(B, n_params)`` holding PDE parameters.
            Parameter values are expected in *physical* units and are
            normalized internally by the caller-supplied scales.

        Returns
        -------
        Tensor of shape ``(B, 1, *spatial)`` holding the normalized next state.
        """

    @property
    def spatial_dims(self) -> int:
        raise NotImplementedError


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Parameter accounting, separating real and complex trainable tensors."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    complex_params = sum(
        p.numel() for p in model.parameters() if p.is_complex()
    )
    real_params = total - complex_params
    return {
        "total": int(total),
        "trainable": int(trainable),
        "real": int(real_params),
        "complex": int(complex_params),
    }
