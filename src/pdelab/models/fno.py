"""Fourier Neural Operator implementations (1D and 2D).

Reference: Li et al., "Fourier Neural Operator for Parametric Partial
Differential Equations", arXiv:2010.08895. This is an independent
implementation in the reference architecture's spirit; the spectral mixing
uses learned complex weights over retained Fourier modes, combined with
pointwise linear (residual) transformations, as described in that paper.

Design notes
------------
* ``rfft`` conventions are used, so only non-negative frequencies are
  stored in the last transformed axis. In 2D, the retained block is
  ``[:modes_x, :modes_y]`` of the ``rfft2`` output, which retains both
  signs in the first (transformed-by-rfft2-as-full) axis and non-negative
  frequencies in the halved axis.
* Weights are stored as complex parameters and are **not** cast to real.
* The number of retained modes is always clipped to the number of
  available Fourier coefficients so Nyquist limits are respected.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TimeAdvanceModel


def _promote_dtype(x: torch.Tensor) -> torch.Tensor:
    """FFTs do not support half precision on CUDA; promote to float32."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


class SpectralConv1d(nn.Module):
    """Learned spectral mixing over the retained 1D Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n = x.shape
        x = _promote_dtype(x)
        x_ft = torch.fft.rfft(x)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            b, self.out_channels, x_ft.shape[-1], device=x.device, dtype=x_ft.dtype
        )
        out_ft[:, :, :m] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :m], self.weights[:, :, :m]
        )
        return torch.fft.irfft(out_ft, n=n)


class SpectralConv2d(nn.Module):
    """Learned spectral mixing over the retained 2D Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, nx, ny = x.shape
        x = _promote_dtype(x)
        x_ft = torch.fft.rfft2(x)
        mx = min(self.modes_x, x_ft.shape[-2])
        my = min(self.modes_y, x_ft.shape[-1])
        out_ft = torch.zeros(
            b, self.out_channels, x_ft.shape[-2], x_ft.shape[-1],
            device=x.device, dtype=x_ft.dtype,
        )
        out_ft[:, :, :mx, :my] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :mx, :my], self.weights[:, :, :mx, :my]
        )
        return torch.fft.irfft2(out_ft, s=(nx, ny))


class FNO1d(TimeAdvanceModel):
    """1D FNO one-step time advance."""

    spatial_dims = 1

    def __init__(
        self,
        modes: int = 16,
        width: int = 16,
        history: int = 1,
        n_params: int = 0,
        n_layers: int = 4,
        out_channels: int = 1,
    ) -> None:
        super().__init__(history=history, n_params=n_params)
        self.width = width
        self.modes = modes
        self.n_layers = n_layers
        # +1 for the normalized grid coordinate channel
        self.fc0 = nn.Linear(history + n_params + 1, width)

        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(
        self, u_hist: torch.Tensor, params: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, h, n = u_hist.shape
        x = u_hist.permute(0, 2, 1)  # [B, N, H]
        if self.n_params:
            if params is None:
                raise ValueError("model expects PDE parameter channels but none given")
            x = torch.cat((x, params.unsqueeze(1).expand(-1, n, -1)), dim=-1)
        grid = torch.linspace(0, 1, n, device=x.device, dtype=x.dtype)
        x = torch.cat((x, grid.reshape(1, n, 1).expand(b, -1, -1)), dim=-1)
        x = self.fc0(x).permute(0, 2, 1)  # [B, W, N]

        for spec, point in zip(self.spectral, self.pointwise):
            x = spec(x) + point(x)
            x = F.gelu(x)

        x = x.permute(0, 2, 1)  # [B, N, W]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)  # [B, N, out]
        return x.permute(0, 2, 1)  # [B, out, N]


class FNO2d(TimeAdvanceModel):
    """2D FNO one-step time advance."""

    spatial_dims = 2

    def __init__(
        self,
        modes_x: int = 8,
        modes_y: int = 8,
        width: int = 16,
        history: int = 1,
        n_params: int = 0,
        n_layers: int = 4,
        out_channels: int = 1,
    ) -> None:
        super().__init__(history=history, n_params=n_params)
        self.width = width
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.n_layers = n_layers
        # +2 for the normalized x and y grid coordinate channels
        self.fc0 = nn.Linear(history + n_params + 2, width)

        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes_x, modes_y) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(
        self, u_hist: torch.Tensor, params: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, h, nx, ny = u_hist.shape
        x = u_hist.permute(0, 2, 3, 1)  # [B, X, Y, H]
        if self.n_params:
            if params is None:
                raise ValueError("model expects PDE parameter channels but none given")
            # (B, P) -> (B, 1, 1, P) -> broadcast over the spatial axes.
            p = params.reshape(b, self.n_params, 1, 1).expand(-1, nx, ny, -1)
            x = torch.cat((x, p), dim=-1)
        gx = torch.linspace(0, 1, nx, device=x.device, dtype=x.dtype)
        gy = torch.linspace(0, 1, ny, device=x.device, dtype=x.dtype)
        gx = gx.reshape(1, nx, 1, 1).expand(b, -1, ny, -1)
        gy = gy.reshape(1, 1, ny, 1).expand(b, nx, -1, -1)
        x = torch.cat((x, gx, gy), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)  # [B, W, X, Y]

        for spec, point in zip(self.spectral, self.pointwise):
            x = spec(x) + point(x)
            x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)  # [B, X, Y, W]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)  # [B, out, X, Y]
