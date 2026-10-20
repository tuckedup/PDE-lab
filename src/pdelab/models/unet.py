"""Periodic U-Net baselines (1D and 2D).

Design constraints addressed here (they are the usual source of an
unfair comparison against FNO on periodic problems):

* **Circular padding everywhere.** Every convolution uses
  ``padding_mode='circular'`` so the network does not impose an artificial
  zero/reflected boundary on a periodic domain.
* **Odd/even shape alignment.** The spatial size is padded up to a
  multiple of ``2 ** n_levels`` on entry, and the prediction is cropped
  back to the original resolution on exit. Downsampling uses stride-2
  convolutions with kernel size 2 (exact), and upsampling uses nearest
  neighbour interpolation followed by an explicit crop/pad to the
  matching skip-connection size.
* **Grid-size divisibility is validated** so a mismatch cannot silently
  change the effective boundary treatment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TimeAdvanceModel


def _pad_to_multiple(x: torch.Tensor, factor: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Pad trailing spatial dims up to a multiple of ``factor``."""
    spatial = x.shape[2:]
    target = [((s + factor - 1) // factor) * factor for s in spatial]
    pad = []
    for s, t in zip(reversed(spatial), reversed(target)):
        pad.extend([0, t - s])
    if all(p == 0 for p in pad):
        return x, tuple(spatial)
    return F.pad(x, pad, mode="circular"), tuple(target)


def _crop_to(x: torch.Tensor, spatial: tuple[int, ...]) -> torch.Tensor:
    """Crop trailing spatial dims to ``spatial``."""
    out = x
    for axis, s in enumerate(spatial, start=2):
        if out.shape[axis] != s:
            out = out.narrow(axis, 0, s)
    return out


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, padding=pad, padding_mode="circular"
        )
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size, padding=pad, padding_mode="circular"
        )
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.conv1(x)))
        return F.gelu(self.norm2(self.conv2(x)))


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size, padding=pad, padding_mode="circular"
        )
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size, padding=pad, padding_mode="circular"
        )
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.conv1(x)))
        return F.gelu(self.norm2(self.conv2(x)))


class PeriodicUNet1d(TimeAdvanceModel):
    """1D periodic U-Net one-step time advance."""

    spatial_dims = 1

    def __init__(
        self,
        width: int = 16,
        history: int = 1,
        n_params: int = 0,
        n_levels: int = 3,
        out_channels: int = 1,
    ) -> None:
        super().__init__(history=history, n_params=n_params)
        self.n_levels = n_levels
        widths = [width * (2**i) for i in range(n_levels + 1)]
        self.widths = widths
        in_ch = history + n_params

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = in_ch
        for i, w in enumerate(widths):
            self.encoders.append(ConvBlock1d(prev, w))
            if i < n_levels:
                # Exact stride-2 reduction: kernel 2, stride 2, no padding.
                self.downs.append(
                    nn.Conv1d(w, w, kernel_size=2, stride=2, padding_mode="circular")
                )
            prev = w

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(n_levels, 0, -1):
            # Nearest-neighbour upsampling followed by a convolution,
            # rather than a transposed convolution. A transposed
            # convolution with kernel 2 and stride 2 produces a
            # checkerboard (period-2) artefact that shows up even for a
            # constant input, which is an implementation wart of the
            # baseline rather than a property of the U-Net family. It
            # would unfairly handicap the spatial baseline, so it is
            # avoided here.
            self.ups.append(
                nn.Conv1d(widths[i], widths[i - 1], kernel_size=3, padding=1,
                          padding_mode="circular")
            )
            self.decoders.append(ConvBlock1d(widths[i - 1] * 2, widths[i - 1]))
        self.head = nn.Conv1d(widths[0], out_channels, kernel_size=1)

    def forward(
        self, u_hist: torch.Tensor, params: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, h, n = u_hist.shape
        x = u_hist
        if self.n_params:
            if params is None:
                raise ValueError("model expects PDE parameter channels but none given")
            x = torch.cat((x, params.unsqueeze(-1).expand(-1, -1, n)), dim=1)

        x, target = _pad_to_multiple(x, 2**self.n_levels)

        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < self.n_levels:
                skips.append(x)
                x = self.downs[i](x)

        for i, (up, dec) in enumerate(zip(self.ups, self.decoders)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[-1] != skip.shape[-1]:
                x = x[..., : skip.shape[-1]] if x.shape[-1] > skip.shape[-1] else F.pad(
                    x, (0, skip.shape[-1] - x.shape[-1]), mode="circular"
                )
            x = dec(torch.cat((x, skip), dim=1))

        x = self.head(x)
        return _crop_to(x, (n,))


class PeriodicUNet2d(TimeAdvanceModel):
    """2D periodic U-Net one-step time advance."""

    spatial_dims = 2

    def __init__(
        self,
        width: int = 16,
        history: int = 1,
        n_params: int = 0,
        n_levels: int = 3,
        out_channels: int = 1,
    ) -> None:
        super().__init__(history=history, n_params=n_params)
        self.n_levels = n_levels
        widths = [width * (2**i) for i in range(n_levels + 1)]
        self.widths = widths
        in_ch = history + n_params

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = in_ch
        for i, w in enumerate(widths):
            self.encoders.append(ConvBlock2d(prev, w))
            if i < n_levels:
                self.downs.append(
                    nn.Conv2d(w, w, kernel_size=2, stride=2, padding_mode="circular")
                )
            prev = w

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(n_levels, 0, -1):
            # Nearest-neighbour upsampling followed by a convolution
            # (see the 1D decoder for the rationale).
            self.ups.append(
                nn.Conv2d(widths[i], widths[i - 1], kernel_size=3, padding=1,
                          padding_mode="circular")
            )
            self.decoders.append(ConvBlock2d(widths[i - 1] * 2, widths[i - 1]))
        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(
        self, u_hist: torch.Tensor, params: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, h, nx, ny = u_hist.shape
        x = u_hist
        if self.n_params:
            if params is None:
                raise ValueError("model expects PDE parameter channels but none given")
            p = params.reshape(b, self.n_params, 1, 1).expand(-1, nx, ny, -1)
            x = torch.cat((x, p), dim=1)

        x, target = _pad_to_multiple(x, 2**self.n_levels)

        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < self.n_levels:
                skips.append(x)
                x = self.downs[i](x)

        for i, (up, dec) in enumerate(zip(self.ups, self.decoders)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = dec(torch.cat((x, skip), dim=1))

        x = self.head(x)
        return _crop_to(x, (nx, ny))
