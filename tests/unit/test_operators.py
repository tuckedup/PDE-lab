"""Tests for the periodic differential operators.

These validate the operators that the physics loss depends on, so a
silent error in them cannot be mistaken for a physics-regularization
effect.

The tests added for the 2026-10-21 audit repair fail against the
pre-repair code:

* the de-aliasing masks were not Hermitian symmetric (the old
  ``mask[:keep] | mask[-keep:]`` construction keeps the modes
  ``0..keep-1`` and ``-keep..-1``, whose mirror images differ),
* the centred first-derivative operator and the centred residual
  operators did not exist,
* the Burgers residual did not project the nonlinear output back onto
  the retained band.
"""

from __future__ import annotations

import numpy as np
import torch

from pdelab.physics.operators import (
    allen_cahn_residual,
    allen_cahn_residual_centered_first_derivative,
    allen_cahn_residual_implicit_euler,
    burgers_residual,
    burgers_residual_centered_first_derivative,
    burgers_residual_implicit_euler,
    d2x_1d,
    ddx_1d,
    dealias_1_2,
    dealias_2_3,
    laplacian_2d,
    mass_1d,
    mass_2d,
    spectral_k_1d,
    spectral_k_2d,
    time_derivative_autograd,
    time_derivative_centered_first_order,
    time_derivative_forward_first_order,
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
    assert mask.sum() == 2 * (n // 3) + 1
    # The Nyquist mode must not survive.
    assert not mask[n // 2]


def test_dealias_mask_is_hermitian_symmetric():
    """The mask must treat the mode ``+k`` and the mode ``-k`` the same.

    The old construction (``mask[:keep] = True; mask[-keep:] = True``)
    kept the modes ``0 .. keep-1`` and ``-keep .. -1``: the mirror of
    ``-keep`` (namely ``+keep``) was dropped, so the mask was *not*
    Hermitian symmetric and a real field did not stay real.
    """
    for n in (32, 48, 64, 96, 128):
        for mask in (dealias_2_3(n), dealias_1_2(n)):
            mirror = mask[(-np.arange(n)) % n]
            assert np.array_equal(mask, mirror), f"n={n}"


def test_dealias_masks_exclude_the_nyquist_mode():
    for n in (32, 64, 128):
        assert not dealias_2_3(n)[n // 2]
        assert not dealias_1_2(n)[n // 2]


def test_dealias_1_2_keeps_half_the_nyquist():
    n = 64
    modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
    assert np.array_equal(dealias_1_2(n), np.abs(modes) <= n // 4)


def test_time_derivative_is_forward_difference():
    u_prev = torch.zeros(1, 8)
    u_next = torch.full((1, 8), 0.5)
    dt = 0.25
    d = time_derivative_autograd(u_next, u_prev, dt)
    assert torch.allclose(d, torch.full((1, 8), 2.0))


def test_forward_and_legacy_time_derivative_agree():
    u_prev = torch.zeros(1, 8)
    u_next = torch.full((1, 8), 0.5)
    assert torch.allclose(
        time_derivative_autograd(u_next, u_prev, 0.25),
        time_derivative_forward_first_order(u_next, u_prev, 0.25),
    )


def test_centred_first_time_derivative_is_second_order():
    """(u_next - u_prev) / (2 dt) must converge at O(dt^2) on a smooth
    trajectory, and it must be evaluated on the interior only.

    The trajectory is the exact spatially uniform Allen-Cahn reaction
    solution, whose time derivative is known in closed form.
    """
    c = 1 / 0.2**2 - 1

    def exact(t):
        return 1 / np.sqrt(1 + c * np.exp(-2 * t))

    def exact_deriv(t):
        return c * np.exp(-2 * t) * (1 + c * np.exp(-2 * t)) ** -1.5

    rows_c = []
    rows_f = []
    for dt in (0.02, 0.01, 0.005):
        times = np.arange(1, int(round(0.2 / dt))) * dt
        traj = torch.as_tensor(exact(times)[None, :, None, None], dtype=torch.float64)
        assert traj.shape[1] == len(times)
        dc = time_derivative_centered_first_order(traj[:, 2:], traj[:, :-2], dt)
        assert dc.shape == (1, traj.shape[1] - 2, 1, 1)
        df = time_derivative_forward_first_order(traj[:, 1:], traj[:, :-1], dt)
        # the centred derivative lives on the interior times t_1..t_{T-2},
        # the forward difference on t_1..t_{T-1}
        ref_c = torch.as_tensor(
            exact_deriv(times[1:-1])[None, :, None, None], dtype=torch.float64
        )
        ref_f = torch.as_tensor(
            exact_deriv(times[:-1])[None, :, None, None], dtype=torch.float64
        )
        rows_c.append(float(torch.sqrt(torch.mean((dc - ref_c) ** 2))))
        rows_f.append(float(torch.sqrt(torch.mean((df - ref_f) ** 2))))
    # centred: O(dt^2) -> the error drops by ~4 per halving
    assert rows_c[-1] / rows_c[0] < 0.2, rows_c
    # forward: O(dt) -> the error only drops by ~2 per halving
    assert rows_f[-1] / rows_f[0] > 0.15, rows_f


def test_allen_cahn_centered_residual_is_second_order_in_time():
    """The residual of the exact reaction solution must converge at
    O(dt^2) with the centred stencil and at O(dt) with the
    implicit-Euler form.  This is the audit's quantity, computed with
    the operator that the evaluation module calls."""
    c = 1 / 0.2**2 - 1

    def exact(t):
        return 1 / np.sqrt(1 + c * np.exp(-2 * t))

    rows_c = []
    rows_i = []
    for dt in (0.02, 0.01, 0.005):
        times = np.arange(1, int(round(0.2 / dt))) * dt
        traj = torch.as_tensor(exact(times)[None, :, None, None], dtype=torch.float64)
        cen = allen_cahn_residual_centered_first_derivative(
            traj, dt=dt, D=0.0, a=1.0, dealias=False
        )
        imp = allen_cahn_residual_implicit_euler(
            traj[:, 1:-1], traj[:, :-2], dt=dt, D=0.0, a=1.0, dealias=False
        )
        rows_c.append(float(torch.sqrt(torch.mean(cen**2))))
        rows_i.append(float(torch.sqrt(torch.mean(imp**2))))
    assert rows_c[-1] / rows_c[0] < 0.2, rows_c
    assert rows_i[-1] / rows_i[0] > 0.15, rows_i


def test_centred_first_derivative_is_exact_for_a_linear_trajectory():
    dt = 0.01
    times = torch.arange(0, 10, dtype=torch.float64) * dt
    traj = (3.0 + 2.0 * times).reshape(1, 10, 1)
    d = time_derivative_centered_first_order(traj[:, 2:], traj[:, :-2], dt)
    assert torch.allclose(d, torch.full_like(d, 2.0), atol=1e-10)


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

    r = burgers_residual(u_next, u, dt=dt, nu=nu, dealias=False)
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


def test_burgers_implicit_euler_residual_is_the_legacy_residual():
    n = 32
    u_prev = torch.randn(1, n, dtype=torch.float64)
    u_next = torch.randn(1, n, dtype=torch.float64)
    a = burgers_residual(u_next, u_prev, dt=0.01, nu=0.01)
    b = burgers_residual_implicit_euler(u_next, u_prev, dt=0.01, nu=0.01)
    assert torch.allclose(a, b)


def test_burgers_residual_projects_the_flux_output_onto_the_retained_band():
    """The old residual filtered the input but left the product's
    out-of-band content in the residual. The corrected operator
    projects the output back, so it agrees with an independently padded
    (alias-free) product."""
    from pdelab.solvers.validation import _padded_product_1d

    n = 64
    rng = np.random.default_rng(11)
    u = torch.as_tensor(rng.standard_normal((1, n)), dtype=torch.float64)
    k = spectral_k_1d(n, dtype=torch.float64)
    mask = torch.as_tensor(dealias_2_3(n), dtype=torch.bool)
    u_filt = torch.fft.ifft(torch.fft.fft(u, dim=-1) * mask, dim=-1).real
    flux_hat = torch.fft.fft(0.5 * u_filt**2, dim=-1) * mask
    flux_x = torch.fft.ifft(1j * k * flux_hat, dim=-1).real
    expected = flux_x - 0.01 * d2x_1d(u, k)
    r = burgers_residual(u, u, dt=1.0, nu=0.01)
    assert torch.allclose(r, expected, atol=1e-10)

    # and the projected product agrees with the padded reference
    exact = 0.5 * _padded_product_1d(u.numpy()[0], 2, n // 3)
    projected = torch.fft.ifft(
        torch.fft.fft(0.5 * u_filt**2, dim=-1) * mask, dim=-1
    ).real.numpy()[0]
    rel = np.linalg.norm(projected - exact) / np.linalg.norm(exact)
    assert rel < 1e-3


def test_burgers_centered_residual_uses_the_interior_support():
    """The centred operator takes the whole trajectory and returns one
    residual per interior snapshot."""
    n = 32
    traj = torch.randn(2, 7, n, dtype=torch.float64)
    r = burgers_residual_centered_first_derivative(traj, dt=0.01, nu=0.01)
    assert r.shape == (2, 5, n)


def test_burgers_centered_residual_matches_the_independent_formula():
    """The centred residual must be the centred time derivative plus the
    spatial operator evaluated on the *centre* snapshot."""
    n = 32
    dt = 0.01
    traj = torch.randn(2, 6, n, dtype=torch.float64)
    r = burgers_residual_centered_first_derivative(traj, dt=dt, nu=0.01, dealias=False)
    centre = traj[:, 1:-1]
    k = spectral_k_1d(n, dtype=torch.float64)
    u_t = (traj[:, 2:] - traj[:, :-2]) / (2 * dt)
    flux_x = torch.fft.ifft(
        1j * k * torch.fft.fft(0.5 * centre**2, dim=-1), dim=-1
    ).real
    expected = u_t + flux_x - 0.01 * d2x_1d(centre, k)
    assert torch.allclose(r, expected, atol=1e-12)


def test_burgers_centered_residual_has_the_expected_time_order():
    """On a spatially uniform trajectory the Burgers residual is just
    the time derivative, so the two time stencils can be compared
    directly: the centred one is O(dt^2), the implicit-Euler one is
    O(dt)."""
    c = 1 / 0.2**2 - 1

    def exact(t):
        return 1 / np.sqrt(1 + c * np.exp(-2 * t))

    def exact_deriv(t):
        return c * np.exp(-2 * t) * (1 + c * np.exp(-2 * t)) ** -1.5

    rows_c = []
    rows_i = []
    for dt in (0.02, 0.01, 0.005):
        times = np.arange(1, int(round(0.2 / dt))) * dt
        traj = torch.as_tensor(exact(times)[None, :, None], dtype=torch.float64)
        cen = burgers_residual_centered_first_derivative(
            traj, dt=dt, nu=0.0, dealias=False
        )
        imp = burgers_residual_implicit_euler(
            traj[:, 1:-1], traj[:, :-2], dt=dt, nu=0.0, dealias=False
        )
        ref_c = torch.as_tensor(
            exact_deriv(times[1:-1])[None, :, None], dtype=torch.float64
        )
        ref_i = torch.as_tensor(
            exact_deriv(times[:-2])[None, :, None], dtype=torch.float64
        )
        rows_c.append(float(torch.sqrt(torch.mean((cen - ref_c) ** 2))))
        rows_i.append(float(torch.sqrt(torch.mean((imp - ref_i) ** 2))))
    assert rows_c[-1] / rows_c[0] < 0.2, rows_c
    assert rows_i[-1] / rows_i[0] > 0.15, rows_i


def test_allen_cahn_residual_zero_for_equilibrium():
    n = 16
    for value in (0.0, 1.0, -1.0):
        u = torch.full((1, n, n), value)
        r = allen_cahn_residual(u, u, dt=0.05, D=0.001, a=1.0)
        assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)


def test_allen_cahn_implicit_euler_residual_is_the_legacy_residual():
    u_prev = torch.randn(1, 8, 8, dtype=torch.float64)
    u_next = torch.randn(1, 8, 8, dtype=torch.float64)
    a = allen_cahn_residual(u_next, u_prev, dt=0.01, D=0.001, a=1.0)
    b = allen_cahn_residual_implicit_euler(u_next, u_prev, dt=0.01, D=0.001, a=1.0)
    assert torch.allclose(a, b)


def test_allen_cahn_centered_residual_uses_the_interior_support():
    n = 8
    traj = torch.randn(2, 7, n, n, dtype=torch.float64)
    r = allen_cahn_residual_centered_first_derivative(traj, dt=0.01, D=0.001, a=1.0)
    assert r.shape == (2, 5, n, n)


def test_allen_cahn_centered_residual_matches_the_independent_formula():
    n = 8
    dt = 0.01
    traj = torch.randn(2, 6, n, n, dtype=torch.float64)
    r = allen_cahn_residual_centered_first_derivative(
        traj, dt=dt, D=0.001, a=1.0, dealias=False
    )
    centre = traj[:, 1:-1]
    kx, ky = spectral_k_2d(n, n, dtype=torch.float64)
    u_t = (traj[:, 2:] - traj[:, :-2]) / (2 * dt)
    lap = laplacian_2d(centre, kx, ky)
    expected = u_t - 0.001 * lap - 1.0 * (centre - centre**3)
    assert torch.allclose(r, expected, atol=1e-12)


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


def test_centered_residual_operators_are_differentiable():
    n = 16
    x = torch.linspace(0, 1, n + 1)[:-1]
    X, Y = torch.meshgrid(x, x, indexing="ij")
    u = (0.5 * torch.cos(2 * torch.pi * X) * torch.cos(2 * torch.pi * Y)).unsqueeze(0)
    traj = torch.cat([u, u, u], dim=1).requires_grad_(True)
    r = burgers_residual_centered_first_derivative(traj, dt=0.01, nu=0.01)
    r.pow(2).mean().backward()
    assert traj.grad is not None
    assert torch.isfinite(traj.grad).all()


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
