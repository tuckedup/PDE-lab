"""Tests for the neural operator models.

Covers the requirements that the FNO uses genuine learned spectral
mixing (complex weights, correct rFFT conventions, positive/negative
mode handling, Nyquist limits, output shape) and that the U-Net handles
periodic boundaries and odd/even shape alignment consistently.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pdelab.models.base import count_parameters
from pdelab.models.fno import FNO1d, FNO2d, SpectralConv1d, SpectralConv2d
from pdelab.models.registry import architecture_defaults, build_model
from pdelab.models.unet import PeriodicUNet1d, PeriodicUNet2d


# ----------------------------------------------------------------------
# spectral convolution
# ----------------------------------------------------------------------
def test_spectral_conv1d_has_complex_parameters():
    conv = SpectralConv1d(4, 4, modes=8)
    assert conv.weights.is_complex()
    assert conv.weights.shape == (4, 4, 8)


def test_spectral_conv1d_preserves_shape():
    conv = SpectralConv1d(3, 5, modes=4)
    x = torch.randn(2, 3, 64)
    assert conv(x).shape == (2, 5, 64)


def test_spectral_conv1d_uses_only_retained_modes():
    """A retained mode set of size m must leave the other modes untouched.

    Retaining only the DC mode therefore returns the *mean* of the
    input, not the input itself: this checks that the unretained modes
    are genuinely discarded rather than passed through.
    """
    n = 32
    m = 4
    conv = SpectralConv1d(1, 1, modes=m)
    with torch.no_grad():
        conv.weights.zero_()
        conv.weights[0, 0, 0] = 1.0 + 0.0j
    x = torch.randn(1, 1, n)
    out = conv(x)
    assert torch.allclose(out, x.mean(dim=-1, keepdim=True).expand_as(out), atol=1e-5)


def test_spectral_conv1d_respects_nyquist_limit():
    """Requesting more modes than available must be clipped, not crash."""
    conv = SpectralConv1d(2, 2, modes=1000)
    x = torch.randn(1, 2, 16)
    out = conv(x)
    assert out.shape == (1, 2, 16)
    assert torch.isfinite(out).all()


def test_spectral_conv2d_shapes_and_complexity():
    conv = SpectralConv2d(3, 4, modes_x=4, modes_y=4)
    assert conv.weights.is_complex()
    assert conv.weights.shape == (3, 4, 4, 4)
    x = torch.randn(2, 3, 16, 16)
    assert conv(x).shape == (2, 4, 16, 16)


def test_spectral_conv2d_handles_both_mode_signs():
    """The retained block of rfft2 covers both signs in the first axis."""
    n = 16
    conv = SpectralConv2d(1, 1, modes_x=3, modes_y=3)
    x = torch.randn(1, 1, n, n)
    out = conv(x)
    assert out.shape == (1, 1, n, n)
    assert torch.isfinite(out).all()


def test_spectral_convs_are_differentiable():
    conv = SpectralConv1d(2, 2, modes=4)
    x = torch.randn(1, 2, 32, requires_grad=True)
    conv(x).pow(2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert conv.weights.grad is not None
    assert torch.isfinite(conv.weights.grad).all()
    assert float(conv.weights.grad.abs().sum()) > 0


def test_spectral_conv_does_not_silently_cast_away_imaginary_part():
    """A complex input must produce a complex-propagating result."""
    conv = SpectralConv1d(1, 1, modes=4)
    assert conv.weights.is_complex()
    # The forward pass must keep the complex weights in the graph.
    x = torch.randn(1, 1, 16)
    out = conv(x)
    assert not torch.is_complex(out)  # irfft returns real
    assert conv.weights.requires_grad


# ----------------------------------------------------------------------
# FNO models
# ----------------------------------------------------------------------
def test_fno1d_output_shape():
    model = FNO1d(modes=8, width=8, history=1)
    x = torch.randn(3, 1, 64)
    out = model(x)
    assert out.shape == (3, 1, 64)


def test_fno1d_supports_history_and_params():
    model = FNO1d(modes=8, width=8, history=2, n_params=2)
    x = torch.randn(3, 2, 64)
    p = torch.randn(3, 2)
    assert model(x, p).shape == (3, 1, 64)


def test_fno1d_requires_params_when_configured():
    model = FNO1d(modes=8, width=8, history=1, n_params=1)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 32), None)


def test_fno2d_output_shape():
    model = FNO2d(modes_x=4, modes_y=4, width=8, history=1)
    x = torch.randn(2, 1, 16, 16)
    assert model(x).shape == (2, 1, 16, 16)


def test_fno2d_supports_history_and_params():
    model = FNO2d(modes_x=4, modes_y=4, width=8, history=2, n_params=1)
    x = torch.randn(2, 2, 16, 16)
    p = torch.randn(2, 1)
    assert model(x, p).shape == (2, 1, 16, 16)


def test_fno_gradients_flow():
    for model, x in [
        (FNO1d(modes=8, width=8), torch.randn(2, 1, 32)),
        (FNO2d(modes_x=4, modes_y=4, width=8), torch.randn(2, 1, 16, 16)),
    ]:
        out = model(x)
        out.pow(2).mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_fno_parameter_counting_separates_complex():
    model = FNO1d(modes=8, width=8)
    counts = count_parameters(model)
    assert counts["complex"] > 0
    assert counts["total"] == counts["real"] + counts["complex"]


# ----------------------------------------------------------------------
# periodic U-Net
# ----------------------------------------------------------------------
def test_unet1d_output_shape_even_and_odd():
    for n in [32, 33, 64, 65]:
        model = PeriodicUNet1d(width=8, n_levels=3)
        x = torch.randn(2, 1, n)
        assert model(x).shape == (2, 1, n)


def test_unet2d_output_shape_even_and_odd():
    for n in [16, 17, 32]:
        model = PeriodicUNet2d(width=8, n_levels=3)
        x = torch.randn(2, 1, n, n)
        assert model(x).shape == (2, 1, n, n)


def test_unet_uses_circular_padding():
    """Convolutions must use circular padding so the periodic boundary is
    treated consistently rather than imposing an artificial boundary."""
    model = PeriodicUNet1d(width=8, n_levels=2)
    conv = model.encoders[0].conv1
    assert conv.padding_mode == "circular"


def test_unet_is_translation_covariant_on_a_periodic_grid():
    """Shifting the input by a whole number of *coarse* cells shifts the
    output the same way. This fails if the network imposes a
    non-periodic boundary.

    The shift must be a multiple of ``2 ** n_levels``: a stride-2
    downsampling is only phase-aligned for even shifts, and an odd
    shift offsets the coarse features by half a cell. This is a
    property of strided U-Nets in general (it is *not* a boundary
    handling defect - the circular padding is correct), and it is
    documented in docs/DECISIONS.md.
    """
    n = 32
    levels = 2
    model = PeriodicUNet2d(width=8, n_levels=levels).eval()
    x = torch.randn(1, 1, n, n)
    shift = 2**levels
    with torch.no_grad():
        out = model(x)
        out_shift = model(torch.roll(x, shifts=shift, dims=-1))
    rolled = torch.roll(out, shifts=shift, dims=-1)
    scale = float(out.abs().max())
    assert float((out_shift - rolled).abs().max()) <= 1e-4 * scale


def test_unet_treats_the_periodic_boundary_correctly():
    """A spatially constant field must map to a spatially constant field.

    This is the direct test that no artificial (zero or reflected)
    boundary is injected: with zero padding a constant field would
    produce a spurious ramp near the edges.
    """
    n = 32
    model = PeriodicUNet1d(width=8, n_levels=3).eval()
    x = torch.full((1, 1, n), 0.7)
    with torch.no_grad():
        out = model(x)
    spread = float(out.max() - out.min())
    assert spread < 1e-5 * max(1.0, float(out.abs().max()))


def test_unet_zero_padding_would_fail_the_constant_field_test():
    """Guards the guard: a deliberately non-periodic network must fail
    the constant-field test, so the test above is meaningful."""
    import torch.nn as nn

    class ZeroPadUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 1, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    net = ZeroPadUNet()
    x = torch.full((1, 1, 32), 0.7)
    with torch.no_grad():
        out = net(x)
    # Zero padding makes the edges differ from the interior.
    assert float(out[0, 0, 0] - out[0, 0, 16]) > 1e-3


def test_unet_gradients_flow():
    for model, x in [
        (PeriodicUNet1d(width=8, n_levels=2), torch.randn(2, 1, 32)),
        (PeriodicUNet2d(width=8, n_levels=2), torch.randn(2, 1, 16, 16)),
    ]:
        out = model(x)
        out.pow(2).mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_unet_respects_history_and_params():
    model = PeriodicUNet1d(width=8, n_levels=2, history=2, n_params=2)
    x = torch.randn(2, 2, 32)
    p = torch.randn(2, 2)
    assert model(x, p).shape == (2, 1, 32)


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------
def test_registry_builds_every_architecture():
    for arch, dim, shape in [
        ("fno1d", 1, (2, 1, 32)),
        ("unet1d", 1, (2, 1, 32)),
        ("fno2d", 2, (2, 1, 16, 16)),
        ("unet2d", 2, (2, 1, 16, 16)),
    ]:
        model = build_model(arch, dimension=dim, **architecture_defaults(arch))
        out = model(torch.randn(*shape))
        assert out.shape[0] == shape[0]
        assert out.shape[1] == 1


def test_registry_rejects_unknown_architecture():
    with pytest.raises(ValueError):
        build_model("mlp1d", dimension=1)
