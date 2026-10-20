"""Tests for the periodic differential operators.

These validate the operators that the physics loss depends on, so a
silent error in them cannot be mistaken for a physics-regularization
effect.
"""

from __future__ import annotations

import numpy as np
import torch

from pdelab.physics.operators import (
    allen_cahn_residual,
    burgers_residual,
    d2x_1d,
    ddx_1d,
    dealias_2_3,
    laplacian_2d,
    mass_1d,
    mass_2d,
    spectral_k_1d,
    spectral_k_2d,
    time_derivative_autograd,
)


def test_spectral_derivative_of_single_mode_is_exact():
    """d/dx of cos(kx) is -k sin(kx), exactly, for a periodic grid.

    The test runs in float64: in float32 the k^2 factor amplifies the
    FFT round-off to ~1e-2 on values of order 350, which is a
    representation limit rather than an operator error.
    """
    n = 64
    k = 3
    x = torch.linspace(0, 1, n + 1, dtype=torch.float64)[:-1]
    u = torch.cos(2 * torch.pi * k * x).unsqueeze(0)
    du = ddx_1d(u)
    exact = -(2 * torch.pi * k) * torch.sin(2 * torch.pi * k * x).unsqueeze(0)
    assert torch.allclose(du, exact, atol=1e-10)


def test_spectral_second_derivative_of_single_mode_is_exact():
    n = 64
    k = 3
    x = torch.linspace(0, 1, n + 1, dtype=torch.float64)[:-1]
    u = torch.cos(2 * torch.pi * k * x).unsqueeze(0)
    d2 = d2x_1d(u)
    exact = -(2 * torch.pi * k) ** 2 * torch.cos(2 * torch.pi * k * x).unsqueeze(0)
    assert torch.allclose(d2, exact, atol=1e-8)


def test_derivative_of_constant_is_zero():
    n = 32
    u = torch.full((1, n), 2.5)
    assert torch.allclose(ddx_1d(u), torch.zeros_like(u), atol=1e-6)
    assert torch.allclose(d2x_1d(u), torch.zeros_like(u), atol=1e-6)


def test_derivative_is_periodic_in_the_meaningful_sense():
    """A periodic function's derivative has zero mean."""
    n = 64
    x = torch.linspace(0, 1, n + 1)[:-1]
    u = torch.cos(2 * torch.pi * x).unsqueeze(0)
    du = ddx_1d(u)
    assert abs(float(du.mean())) < 1e-6


def test_2d_laplacian_of_single_mode_is_exact():
    nx = ny = 32
    kx, ky = 2, 3
    x = torch.linspace(0, 1, nx + 1)[:-1]
    y = torch.linspace(0, 1, ny + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")
    u = (torch.cos(2 * torch.pi * kx * X) * torch.cos(2 * torch.pi * ky * Y)).unsqueeze(0)
    lap = laplacian_2d(u)
    exact = -((2 * torch.pi * kx) ** 2 + (2 * torch.pi * ky) ** 2) * u
    assert torch.allclose(lap, exact, atol=1e-2)


def test_2d_laplacian_of_constant_is_zero():
    u = torch.full((1, 8, 8), 3.0)
    assert torch.allclose(laplacian_2d(u), torch.zeros_like(u), atol=1e-8)


def test_wavenumbers_match_nfft():
    n = 32
    k = spectral_k_1d(n)
    expected = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    assert np.allclose(k.numpy(), expected)


def test_2d_wavenumbers_match_nfft():
    nx, ny = 8, 12
    kx, ky = spectral_k_2d(nx, ny)
    ex = 2 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)
    ey = 2 * np.pi * np.fft.fftfreq(ny, d=1.0 / ny)
    assert np.allclose(kx[:, 0].numpy(), ex)
    assert np.allclose(ky[0, :].numpy(), ey)


def test_dealias_mask_keeps_two_thirds():
    n = 64
    mask = dealias_2_3(n)
    assert mask.sum() == 2 * (n // 3)
    # The Nyquist mode must not survive.
    assert not mask[n // 2]


def test_time_derivative_is_forward_difference():
    u_prev = torch.zeros(1, 8)
    u_next = torch.full((1, 8), 0.5)
    dt = 0.25
    d = time_derivative_autograd(u_next, u_prev, dt)
    assert torch.allclose(d, torch.full((1, 8), 2.0))


def test_burgers_residual_matches_an_independent_computation():
    """The residual must match an independently written expression.

    The residual is evaluated on the *predicted* state with the actual
    observation time step, which is what the physics regulariser uses.
    """
    nu = 0.01
    n = 64
    t = 0.05
    x = torch.linspace(0, 1, n + 1, dtype=torch.float64)[:-1]
    u = (torch.exp(torch.tensor(-t, dtype=torch.float64))
         * torch.cos(2 * torch.pi * x)).unsqueeze(0)
    dt = 1e-6
    u_next = (torch.exp(torch.tensor(-(t + dt), dtype=torch.float64))
              * torch.cos(2 * torch.pi * x)).unsqueeze(0)

    r = burgers_residual(u_next, u, dt=dt, nu=nu)
    # Independent computation with the spectral operators.
    k = spectral_k_1d(n, dtype=torch.float64)
    u_t = (u_next - u) / dt
    flux_hat = torch.fft.fft(0.5 * u_next**2, dim=-1)
    flux_x = torch.fft.ifft(1j * k * flux_hat, dim=-1).real
    u_xx = d2x_1d(u_next, k)
    expected = u_t + flux_x - nu * u_xx
    assert torch.allclose(r, expected, atol=1e-8)


def test_burgers_residual_is_zero_for_a_constant_state():
    """A constant state has zero residual for any dt (u_t = 0, all
    spatial derivatives vanish)."""
    n = 32
    u = torch.full((1, n), 0.7)
    r = burgers_residual(u, u, dt=0.01, nu=0.01)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)


def test_allen_cahn_residual_zero_for_equilibrium():
    n = 16
    for value in (0.0, 1.0, -1.0):
        u = torch.full((1, n, n), value)
        r = allen_cahn_residual(u, u, dt=0.05, D=0.001, a=1.0)
        assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)


def test_residual_operators_are_differentiable():
    """Gradients must flow through the residual to the input field."""
    n = 32
    x = torch.linspace(0, 1, n + 1)[:-1]
    u = (0.5 * torch.cos(2 * torch.pi * x)).unsqueeze(0).requires_grad_(True)
    u_prev = u.detach().clone()
    r = burgers_residual(u, u_prev, dt=0.01, nu=0.01)
    r.pow(2).mean().backward()
    assert u.grad is not None
    assert torch.isfinite(u.grad).all()
    assert float(u.grad.abs().sum()) > 0


def test_2d_residual_operators_are_differentiable():
    n = 16
    x = torch.linspace(0, 1, n + 1)[:-1]
    X, Y = torch.meshgrid(x, x, indexing="ij")
    u = (0.5 * torch.cos(2 * torch.pi * X) * torch.cos(2 * torch.pi * Y)).unsqueeze(0)
    u = u.requires_grad_(True)
    u_prev = u.detach().clone()
    r = allen_cahn_residual(u, u_prev, dt=0.05, D=0.001, a=1.0)
    r.pow(2).mean().backward()
    assert u.grad is not None
    assert torch.isfinite(u.grad).all()
    assert float(u.grad.abs().sum()) > 0


def test_mass_helpers():
    u = torch.full((1, 8), 0.5)
    assert torch.allclose(mass_1d(u), torch.full((1,), 0.5))
    u2 = torch.full((1, 4, 4), 0.25)
    assert torch.allclose(mass_2d(u2), torch.full((1,), 0.25))
