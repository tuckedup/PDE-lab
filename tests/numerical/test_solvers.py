"""Numerical tests for the reference solvers.

These are tiny, fast checks with meaningful tolerances. They are
suitable for CI and do not need a GPU.
"""

from __future__ import annotations

import numpy as np

from pdelab.solvers.allen_cahn import AllenCahn2DSpectral
from pdelab.solvers.burgers import Burgers1DFiniteVolume, ssprk3_step
from pdelab.solvers.burgers_weno import Burgers1DWeno5
from pdelab.solvers.validation import validate_allen_cahn, validate_burgers


# ----------------------------------------------------------------------
# Burgers
# ----------------------------------------------------------------------
def test_burgers_constant_state_is_stationary():
    s = Burgers1DWeno5(n=32, nu=0.01)
    u = np.full(32, 0.3)
    out = s.solve(u, dt=1e-3, num_steps=20)
    assert np.allclose(out[-1], u, atol=1e-12)


def test_burgers_mean_is_conserved():
    s = Burgers1DWeno5(n=64, nu=0.01)
    x = s.x
    u0 = 0.8 * np.cos(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x + 0.5)
    out = s.solve(u0, dt=2e-3, num_steps=200)
    means = out.mean(axis=1)
    assert np.max(np.abs(means - means[0])) < 1e-12


def test_burgers_energy_is_non_increasing():
    s = Burgers1DWeno5(n=64, nu=0.01)
    x = s.x
    u0 = 0.8 * np.cos(2 * np.pi * x)
    out = s.solve(u0, dt=2e-3, num_steps=200)
    energy = 0.5 * np.sum(out**2, axis=1) / out.shape[1]
    assert np.all(np.diff(energy) <= 1e-12)


def test_burgers_endpoint_is_not_duplicated():
    s = Burgers1DWeno5(n=32, nu=0.01)
    assert len(s.x) == 32
    assert s.x[0] == 0.0
    assert not np.isclose(s.x[-1], 1.0)
    assert np.isclose(s.x[1] - s.x[0], 1.0 / 32)


def test_burgers_diffusion_only_decays_a_fourier_mode():
    s = Burgers1DWeno5(n=64, nu=0.01)
    u = np.cos(2 * np.pi * s.x)
    t = 0.05
    dt = 1e-4
    for _ in range(int(t / dt)):
        u = s.diffusion_only_step(u, dt)
    exact = np.exp(-0.01 * (2 * np.pi) ** 2 * t) * np.cos(2 * np.pi * s.x)
    assert np.max(np.abs(u - exact)) < 5e-4


def test_ssprk3_integrator_is_consistent():
    """The time integrator must reproduce the exact solution of a linear
    ODE. An inconsistent stage/weight pairing is silently first order and
    would not be caught by a self-convergence test."""
    a = 0.7
    u = np.ones(8)
    dt = 1e-3
    for _ in range(1000):
        u = ssprk3_step(u, dt, lambda v: a * v)
    assert np.max(np.abs(u - np.exp(a * dt * 1000))) < 1e-9


def test_burgers_spatial_operator_is_second_order():
    """The discrete RHS must converge at second order to the exact
    continuous RHS. This is the check that detects a mis-indexed flux
    divergence."""
    n_fine = 2048
    x_fine = np.arange(n_fine) / n_fine
    u_fine = 0.8 * np.cos(2 * np.pi * x_fine) + 0.3 * np.cos(4 * np.pi * x_fine + 0.5)
    k = 2 * np.pi * np.fft.fftfreq(n_fine, d=1.0 / n_fine)
    uh = np.fft.fft(u_fine)
    uxx = np.fft.ifft(-(k**2) * uh).real
    fluxx = np.fft.ifft(1j * k * np.fft.fft(0.5 * u_fine**2)).real
    rhs_exact = -fluxx + 0.01 * uxx

    errors = []
    for n in [32, 64, 128]:
        s = Burgers1DWeno5(n=n, nu=0.01)
        u = 0.8 * np.cos(2 * np.pi * s.x) + 0.3 * np.cos(4 * np.pi * s.x + 0.5)
        r = s._rhs(u)
        errors.append(
            np.linalg.norm(r - rhs_exact[:: n_fine // n]) / np.linalg.norm(rhs_exact)
        )
    orders = [np.log2(a / b) for a, b in zip(errors[:-1], errors[1:])]
    assert all(o > 1.5 for o in orders), orders


def test_weno5_and_muscl_agree_in_the_smooth_regime():
    a = Burgers1DWeno5(n=64, nu=0.01)
    b = Burgers1DFiniteVolume(n=64, nu=0.01)
    x = a.x
    u0 = 0.8 * np.cos(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x + 0.5)
    t = 0.02
    ua = a.solve(u0, t / 100, 100)[-1]
    ub = b.solve(u0, t / 100, 100)[-1]
    rel = np.linalg.norm(ua - ub) / np.linalg.norm(ub)
    assert rel < 1e-2


def test_burgers_full_validation_suite_passes():
    result = validate_burgers()
    failed = [c["name"] for c in result["checks"] if not c["passed"]]
    assert not failed, f"failing Burgers checks: {failed}"


# ----------------------------------------------------------------------
# Allen-Cahn
# ----------------------------------------------------------------------
def test_allen_cahn_equilibria_are_stationary():
    s = AllenCahn2DSpectral(nx=16, ny=16, D=0.001, a=1.0)
    for value in (0.0, 1.0, -1.0):
        u = np.full((16, 16), value)
        out = s.solve(u, dt=0.01, num_steps=50)
        assert np.max(np.abs(out[-1] - u)) < 1e-10


def test_allen_cahn_reaction_matches_the_exact_quartic():
    """With D = 0 the equation is an ODE with a closed-form solution."""
    a = 1.0
    u0 = 0.5
    s = AllenCahn2DSpectral(nx=4, ny=4, D=0.0, a=a)
    t = 1.0
    out = s.solve(np.full((4, 4), u0), dt=1e-3, num_steps=int(t / 1e-3))
    exact = u0 / np.sqrt(u0**2 + (1 - u0**2) * np.exp(-2 * a * t))
    assert np.max(np.abs(out[-1] - exact)) < 1e-4


def test_allen_cahn_free_energy_decreases():
    s = AllenCahn2DSpectral(nx=32, ny=32, D=0.001, a=1.0)
    X, Y = np.meshgrid(s.x, s.x, indexing="ij")
    u = 0.5 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    e0 = s.energy(u)
    for _ in range(100):
        u = s.step(u, 0.01)
    assert s.energy(u) < e0


def test_allen_cahn_mass_is_not_conserved():
    """Guards against a false conservation constraint being applied."""
    s = AllenCahn2DSpectral(nx=32, ny=32, D=0.001, a=1.0)
    X, Y = np.meshgrid(s.x, s.x, indexing="ij")
    u0 = 0.3 + 0.4 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    out = s.solve(u0, dt=0.005, num_steps=200)
    means = out.mean(axis=(1, 2))
    assert np.max(np.abs(means - means[0])) > 1e-6


def test_allen_cahn_full_validation_suite_passes():
    result = validate_allen_cahn()
    failed = [c["name"] for c in result["checks"] if not c["passed"]]
    assert not failed, f"failing Allen-Cahn checks: {failed}"
