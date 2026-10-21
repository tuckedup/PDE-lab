"""Measure the spectral coverage of the 2D FNO spectral convolution.

The probe answers one question: *does the spectral convolution mix the
low frequencies of the **first** (non-halved) axis on both sides of the
wave-number axis, or only the positive ones?*  ``torch.fft.rfft2``
keeps both signs of that axis, so an operator that retains only the
positive block drops one half of every Hermitian pair.

The probe works against both the old implementation (a single
``weights`` tensor that covers the positive block only) and the
repaired one (a non-negative and a negative block), so the before and
after numbers are produced by the same script.

Usage::

    .venv/bin/python campaigns/performance/fno2d_spectral_probe.py
"""

from __future__ import annotations

import json
import sys

import torch

from pdelab.models.fno import SpectralConv2d


def _is_repaired(conv: SpectralConv2d) -> bool:
    return hasattr(conv, "weights_nonnegative")


def _set_identity(
    conv: SpectralConv2d, nx: int, my: int, *, block: str = "both"
) -> dict[str, int]:
    """Set the weights to the identity on the retained band.

    ``block`` selects the non-negative block, the negative block or
    both.  The old implementation only has the non-negative block.
    """
    if not _is_repaired(conv):
        mx = min(conv.modes_x, nx)
        with torch.no_grad():
            conv.weights.zero_()
            conv.weights[0, 0, :mx, :my] = 1.0 + 0.0j
        return {"n_nonneg": mx, "n_neg": 0}

    n_nonneg, n_neg = conv.band_rows(nx)
    with torch.no_grad():
        conv.weights_nonnegative.zero_()
        conv.weights_negative.zero_()
        if block in ("both", "nonnegative"):
            conv.weights_nonnegative[0, 0, :n_nonneg, :my] = 1.0 + 0.0j
        if block in ("both", "negative"):
            conv.weights_negative[0, 0, :n_neg, :my] = 1.0 + 0.0j
    return {"n_nonneg": n_nonneg, "n_neg": n_neg}


def _relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    num = float((out - ref).double().pow(2).sum())
    den = float(ref.double().pow(2).sum())
    return (num / den) ** 0.5 if den > 0 else float("nan")


def _cosine(nx: int, ny: int, kx: int, ky: int = 0) -> torch.Tensor:
    x = torch.arange(nx, dtype=torch.float64).reshape(nx, 1) / nx
    y = torch.arange(ny, dtype=torch.float64).reshape(1, ny) / ny
    field = torch.cos(2.0 * torch.pi * (kx * x + ky * y))
    return field.reshape(1, 1, nx, ny).to(torch.float32)


def _random_field(nx: int, ny: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, nx, ny, generator=g)


def _band_limited_field(
    nx: int, ny: int, modes_x: int, modes_y: int, seed: int = 0
) -> torch.Tensor:
    """A real field whose spectrum lies inside the retained band."""
    my = min(modes_y, ny // 2 + 1)
    x = torch.arange(nx, dtype=torch.float64).reshape(nx, 1) / nx
    y = torch.arange(ny, dtype=torch.float64).reshape(1, ny) / ny
    field = torch.zeros(nx, ny, dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    for kx in range(min(modes_x, nx // 2 + 1)):
        for ky in range(my):
            amp = float(torch.rand(1, generator=gen))
            field += amp * torch.cos(2.0 * torch.pi * (kx * x + ky * y))
    return field.reshape(1, 1, nx, ny).to(torch.float32)


def probe(nx: int = 32, ny: int = 32, modes_x: int = 8, modes_y: int = 8) -> dict:
    conv = SpectralConv2d(1, 1, modes_x=modes_x, modes_y=modes_y)
    my = min(modes_y, ny // 2 + 1)
    repaired = _is_repaired(conv)

    info: dict = {
        "implementation": "repaired" if repaired else "legacy (positive block only)",
        "grid": [nx, ny],
        "modes": [modes_x, modes_y],
    }
    if repaired:
        info["retained_rows"] = conv.retained_row_indices(nx)
        info["retained_wave_numbers"] = conv.retained_wave_numbers(nx)

    # 1. identity weights on the whole retained band vs. a random field
    counts = _set_identity(conv, nx, my, block="both")
    info["rows_used"] = counts
    field = _random_field(nx, ny)
    out = conv(field)
    info["identity_vs_random_field_relative_error"] = _relative_error(out, field)

    # 1b. the same, but for a field whose spectrum lies *inside* the
    # retained band: the identity action must reproduce it exactly.
    banded = _band_limited_field(nx, ny, modes_x, modes_y, seed=7)
    out_banded = conv(banded)
    info["identity_vs_band_limited_field_relative_error"] = _relative_error(
        out_banded, banded
    )

    # 2. identity weights vs. a single cosine (kx = +2)
    cosine = _cosine(nx, ny, 2)
    out_cos = conv(cosine)
    info["identity_vs_cosine_kx2_relative_error"] = _relative_error(out_cos, cosine)

    # 3. the negative block on its own (repaired implementation only)
    if repaired:
        _set_identity(conv, nx, my, block="negative")
        out_neg = conv(cosine)
        info["negative_block_only_vs_cosine_relative_error"] = _relative_error(
            out_neg, cosine
        )
        info["negative_block_only_amplitude_ratio"] = float(
            out_neg.abs().max() / cosine.abs().max()
        )
        _set_identity(conv, nx, my, block="nonnegative")
        out_pos = conv(cosine)
        info["nonnegative_block_only_vs_cosine_relative_error"] = _relative_error(
            out_pos, cosine
        )
        info["nonnegative_block_only_amplitude_ratio"] = float(
            out_pos.abs().max() / cosine.abs().max()
        )

    # 4. the spectrum of the identity action must stay Hermitian
    if repaired:
        _set_identity(conv, nx, my, block="both")
        out = conv(field)
        ft = torch.fft.rfft2(out.double())
        ft_ref = torch.fft.rfft2(field.double())
        n_nonneg, n_neg = conv.band_rows(nx)
        expected = torch.zeros_like(ft_ref)
        expected[:, :, :n_nonneg, :my] = ft_ref[:, :, :n_nonneg, :my]
        expected[:, :, nx - n_neg :, :my] = ft_ref[:, :, nx - n_neg :, :my]
        info["hermitian_spectrum_max_error"] = float((ft - expected).abs().max())
    return info


def main() -> int:
    results = []
    for nx, ny in [(32, 32), (16, 16), (15, 15), (17, 17)]:
        results.append(probe(nx=nx, ny=ny))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
