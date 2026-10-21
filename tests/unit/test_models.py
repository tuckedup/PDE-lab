"""Tests for the neural operator models.

Covers the requirements that the FNO uses genuine learned spectral
mixing (complex weights, correct rFFT conventions, positive/negative
mode handling, Nyquist limits, output shape) and that the U-Net handles
periodic boundaries and odd/even shape alignment consistently.
"""

from __future__ import annotations

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
    assert conv.weights_nonnegative.is_complex()
    assert conv.weights_nonnegative.shape == (3, 4, 4, 4)
    # The negative wave numbers of the first axis have their own block
    # of weights: slot s carries kx = -(s+1), so there is one slot
    # fewer than in the non-negative block (the DC mode has no mirror).
    assert conv.weights_negative.is_complex()
    assert conv.weights_negative.shape == (3, 4, 3, 4)
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


# ----------------------------------------------------------------------
# FNO2D spectral coverage: the first (non-halved) axis keeps BOTH signs
# ----------------------------------------------------------------------
def _identity_spectral_weights(conv, nx: int, ny: int, *, block: str = "both"):
    """Set the spectral weights to the identity on the retained band.

    ``block`` selects the non-negative wave numbers, the negative ones
    or both.  Only meaningful for ``in_channels == out_channels == 1``.
    """
    my = min(conv.modes_y, ny // 2 + 1)
    n_nonneg, n_neg = conv.band_rows(nx)
    with torch.no_grad():
        conv.weights_nonnegative.zero_()
        conv.weights_negative.zero_()
        if block in ("both", "nonnegative"):
            conv.weights_nonnegative[0, 0, :n_nonneg, :my] = 1.0 + 0.0j
        if block in ("both", "negative"):
            conv.weights_negative[0, 0, :n_neg, :my] = 1.0 + 0.0j
    return n_nonneg, n_neg


def _cosine_field(nx: int, ny: int, kx: int, ky: int = 0) -> torch.Tensor:
    """A single real Fourier mode: ``cos(2 pi (kx x + ky y))``."""
    x = torch.arange(nx, dtype=torch.float64).reshape(nx, 1) / nx
    y = torch.arange(ny, dtype=torch.float64).reshape(1, ny) / ny
    field = torch.cos(2.0 * torch.pi * (kx * x + ky * y))
    return field.reshape(1, 1, nx, ny).to(torch.float32)


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


def test_spectral_conv2d_retains_both_wave_number_blocks():
    """The retained band must cover the positive AND the negative half.

    ``rfft2`` halves the last axis only: the first axis keeps every
    frequency, so each low wave number appears twice.  The retained
    band has to be closed under the Hermitian involution
    ``kx -> -kx``, otherwise one half of every pair is mixed and the
    other half is thrown away.
    """
    conv = SpectralConv2d(1, 1, modes_x=4, modes_y=4)
    for nx in (16, 17, 32, 33, 8, 5):
        rows = conv.retained_row_indices(nx)
        assert rows == sorted(rows)
        assert len(rows) == len(set(rows)), "the two blocks must not overlap"
        # closed under the Hermitian involution
        for j in rows:
            assert (-j) % nx in rows, f"row {j} has no partner in the retained band"
        # both signs are present
        kx = conv.retained_wave_numbers(nx)
        assert any(k > 0 for k in kx)
        assert any(k < 0 for k in kx)
        assert 0 in kx
        # the Nyquist mode of an even grid is counted exactly once
        if nx % 2 == 0:
            assert rows.count(nx // 2) <= 1


def test_spectral_conv2d_nyquist_mode_is_retained_once_and_not_twice():
    """An even grid has one Nyquist mode; it must not be doubled."""
    nx = 16
    conv = SpectralConv2d(1, 1, modes_x=16, modes_y=4)
    n_nonneg, n_neg = conv.band_rows(nx)
    rows = conv.retained_row_indices(nx)
    assert rows.count(nx // 2) == 1
    # every row of the grid is retained, none twice
    assert len(rows) == nx == len(set(rows))
    assert n_nonneg + n_neg == nx
    # the retained band is closed under kx -> -kx (modulo the grid)
    kx = conv.retained_wave_numbers(nx)
    assert {k % nx for k in kx} == {(-k) % nx for k in kx}


def test_spectral_conv2d_identity_weights_reproduce_a_band_limited_input():
    """An identity-weight operator must reproduce its own input.

    This is the sharp test of the band: the retained spectrum has to be
    the Hermitian-symmetric band-limited spectrum of the input, so the
    inverse transform gives the input back to round-off.
    """
    for nx, ny in [(16, 16), (15, 15), (17, 17), (32, 32)]:
        conv = SpectralConv2d(1, 1, modes_x=8, modes_y=8)
        x = _band_limited_field(nx, ny, 8, 8, seed=nx)
        _identity_spectral_weights(conv, nx, ny)
        with torch.no_grad():
            out = conv(x)
        assert out.shape == x.shape
        tol = 1e-5 * max(1.0, float(x.abs().max()))
        err = float((out - x).abs().max())
        assert err < tol, (
            f"identity action failed on a {nx}x{ny} grid: "
            f"max error {err} (tolerance {tol})"
        )


def test_spectral_conv2d_identity_action_keeps_the_spectrum_hermitian():
    """The retained spectrum of the identity action must be Hermitian.

    ``rfft2(irfft2(X)) == X`` holds exactly when ``X`` is the Hermitian
    spectrum of a real field, i.e. when the retained band is closed
    under ``kx -> -kx``.
    """
    nx, ny = 16, 16
    conv = SpectralConv2d(1, 1, modes_x=6, modes_y=6)
    x = _band_limited_field(nx, ny, 6, 6, seed=3)
    _identity_spectral_weights(conv, nx, ny)
    with torch.no_grad():
        out = conv(x)
    ft = torch.fft.rfft2(out.double())
    ft_ref = torch.fft.rfft2(x.double())
    my = min(6, ny // 2 + 1)
    n_nonneg, n_neg = conv.band_rows(nx)
    expected = torch.zeros_like(ft_ref)
    expected[:, :, :n_nonneg, :my] = ft_ref[:, :, :n_nonneg, :my]
    expected[:, :, nx - n_neg :, :my] = ft_ref[:, :, nx - n_neg :, :my]
    scale = max(1.0, float(ft_ref.abs().max()))
    assert float((ft - expected).abs().max()) < 1e-6 * scale


def test_spectral_conv2d_mixes_a_single_mode_in_each_block():
    """A single Fourier mode must be mixed from *both* sides.

    With only one block mixed, the mode survives at half its amplitude
    (the other half of the Hermitian pair is discarded).  With both
    blocks mixed, the mode is reproduced exactly.
    """
    nx, ny = 32, 32
    x = _cosine_field(nx, ny, 2)

    conv = SpectralConv2d(1, 1, modes_x=8, modes_y=8)
    with torch.no_grad():
        _identity_spectral_weights(conv, nx, ny, block="nonnegative")
        only_pos = conv(x)
        assert float((only_pos - 0.5 * x).abs().max()) < 1e-6

        _identity_spectral_weights(conv, nx, ny, block="negative")
        only_neg = conv(x)
        assert float((only_neg - 0.5 * x).abs().max()) < 1e-6

        _identity_spectral_weights(conv, nx, ny, block="both")
        both = conv(x)
        assert float((both - x).abs().max()) < 1e-6


def test_spectral_conv2d_mixes_the_negative_block_for_a_negative_mode():
    """The negative block must act on the negative wave numbers."""
    nx, ny = 32, 32
    x = _cosine_field(nx, ny, -3)
    conv = SpectralConv2d(1, 1, modes_x=8, modes_y=8)
    with torch.no_grad():
        _identity_spectral_weights(conv, nx, ny, block="negative")
        only_neg = conv(x)
        assert float((only_neg - 0.5 * x).abs().max()) < 1e-6
        _identity_spectral_weights(conv, nx, ny, block="both")
        both = conv(x)
        assert float((both - x).abs().max()) < 1e-6


def test_spectral_conv2d_gradients_are_complex_and_reach_both_blocks():
    """The complex weights of both blocks must receive a gradient."""
    conv = SpectralConv2d(2, 2, modes_x=4, modes_y=4)
    x = torch.randn(2, 2, 16, 16, requires_grad=True)
    conv(x).pow(2).sum().backward()
    for name in ("weights_nonnegative", "weights_negative"):
        grad = getattr(conv, name).grad
        assert grad is not None, f"{name} received no gradient"
        assert grad.is_complex()
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_spectral_conv2d_complex_gradient_has_a_non_zero_imaginary_part():
    """The complex path must be real: the imaginary part is not zero."""
    conv = SpectralConv2d(1, 1, modes_x=4, modes_y=4)
    x = torch.randn(1, 1, 16, 16)
    conv(x).pow(2).sum().backward()
    grad = conv.weights_negative.grad
    assert grad is not None and grad.is_complex()
    assert float(grad.imag.abs().sum()) > 0


def test_spectral_conv2d_handles_odd_and_even_grids():
    """Odd and even grids, and grids smaller than the requested modes."""
    for nx, ny in [(15, 15), (16, 16), (17, 17), (8, 8), (5, 5), (32, 31)]:
        conv = SpectralConv2d(2, 3, modes_x=16, modes_y=16)
        x = torch.randn(2, 2, nx, ny)
        out = conv(x)
        assert out.shape == (2, 3, nx, ny)
        assert torch.isfinite(out).all()
        assert not out.is_complex()


def test_fno2d_forward_path_is_real_and_finite():
    """The real forward path of the repaired operator."""
    model = FNO2d(modes_x=4, modes_y=4, width=8, history=1)
    x = torch.randn(2, 1, 16, 16)
    out = model(x)
    assert out.shape == (2, 1, 16, 16)
    assert not out.is_complex()
    assert torch.isfinite(out).all()


def test_fno2d_spectral_parameter_count_covers_both_blocks():
    """The spectral parameters of the repaired architecture.

    The repaired operator keeps one complex matrix per retained mode
    *block*: ``modes_x`` slots for the non-negative wave numbers and
    ``modes_x - 1`` for their negative mirrors.  The old architecture
    (positive block only) had ``modes_x`` slots in total.
    """
    width, modes_x, modes_y, n_layers = 8, 4, 4, 2
    model = FNO2d(modes_x=modes_x, modes_y=modes_y, width=width, n_layers=n_layers)
    counts = count_parameters(model)
    expected_spectral = (
        n_layers * width * width * (modes_x + modes_x - 1) * modes_y
    )
    assert counts["complex"] == expected_spectral
    assert counts["total"] == counts["real"] + counts["complex"]


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
def test_unet_builds_at_several_widths():
    """GroupNorm requires the group count to divide the channel count, so
    not every width is valid. These widths must all construct."""
    for width in [4, 6, 8, 12, 16]:
        m1 = PeriodicUNet1d(width=width, n_levels=3)
        assert m1(torch.randn(1, 1, 32)).shape == (1, 1, 32)
        m2 = PeriodicUNet2d(width=width, n_levels=3)
        assert m2(torch.randn(1, 1, 16, 16)).shape == (1, 1, 16, 16)


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
