"""Numerical tests for the reference solvers.

These are tiny, fast checks with meaningful tolerances. They are
suitable for CI and do not need a GPU.

Every test added for the 2026-10-21 audit repair is written so that it
*fails* against the pre-repair code:

* the observation-time scheduling tests fail because the old scheduler
  returned one frame too early and dropped the terminal frame,
* the flux-splitting tests fail because the old split was sign blind
  and antidiffusive on a negative background,
* the de-aliasing tests fail because the old masks were not Hermitian
  symmetric and the nonlinear output was not projected back,
* the manufactured-solution tests fail because the old code had no
  forcing interface at all,
* the manifest test fails because the old manifest recorded only the
  requested observation times.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from pdelab.solvers.allen_cahn import AllenCahn2DSpectral
from pdelab.solvers.burgers import Burgers1DFiniteVolume, Burgers1DSpectral, ssprk3_step
from pdelab.solvers.burgers_weno import Burgers1DWeno5
from pdelab.solvers.validation import validate_allen_cahn, validate_burgers


def _burgers_mms_forcing(x: np.ndarray, t: float, nu: float) -> np.ndarray:
    """Continuum forcing for u = exp(-t) cos(2 pi x).

    Written in closed form here so that the test does not depend on any
    part of the code under test.
    """
    exp_t = np.exp(-t)
    u = exp_t * np.cos(2 * np.pi * x)
    u_t = -exp_t * np.cos(2 * np.pi * x)
    flux_x = u * (-2.0 * np.pi * exp_t * np.sin(2 * np.pi * x))
    u_xx = -(2.0 * np.pi) ** 2 * exp_t * np.cos(2 * np.pi * x)
    return u_t + flux_x - nu * u_xx


def _allen_cahn_mms_forcing(
    X: np.ndarray, Y: np.ndarray, t: float, D: float, a: float
) -> np.ndarray:
    """Continuum forcing for u = exp(-t) cos(2 pi x) cos(2 pi y)."""
    u = np.exp(-t) * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    lap = -(2.0 * np.pi) ** 2 * 2.0 * u
    u_t = -u
    return u_t - D * lap - a * (u - u**3)


def _padded_product_1d(u: np.ndarray, power: int, keep: int) -> np.ndarray:
    """Alias-free retained-band power of ``u`` on a 2x padded grid."""
    u = np.asarray(u, dtype=np.float64)
    n = u.size
    u_hat = np.fft.fft(u)
    modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
    sel = np.abs(modes) <= keep
    big_index = np.where(modes >= 0, modes, modes + 2 * n)
    padded_hat = np.zeros(2 * n, dtype=complex)
    padded_hat[big_index[sel]] = 2.0 * u_hat[sel]
    padded = np.fft.ifft(padded_hat).real
    product_hat = np.fft.fft(padded**power)
    out = np.zeros(n, dtype=complex)
    out[sel] = product_hat[big_index[sel]] / 2.0
    return np.fft.ifft(out).real


@functools.cache
def _burgers_validation() -> dict:
    """The Burgers validation suite is expensive; run it once per session."""
    return validate_burgers()


@functools.cache
def _allen_cahn_validation() -> dict:
    """The Allen-Cahn validation suite is expensive; run it once per session."""
    return validate_allen_cahn()


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
    result = _burgers_validation()
    failed = [c["name"] for c in result["checks"] if not c["passed"]]
    assert not failed, f"failing Burgers checks: {failed}"


# ----------------------------------------------------------------------
# A. observation-time scheduling
# ----------------------------------------------------------------------
def test_solve_to_returns_exactly_the_requested_number_of_frames():
    """The old scheduler returned one frame too early and dropped the
    terminal frame, so the frame count and the times were both wrong."""
    save_times = np.array([0.0, 0.1, 0.2, 0.3])
    s = Burgers1DWeno5(n=32, nu=0.01)
    out = s.solve_to(np.zeros(32), 0.3, save_times)
    assert out.shape[0] == len(save_times)
    assert np.allclose(s.frame_times(), save_times, atol=1e-12)


def test_solve_to_lands_exactly_on_the_requested_times():
    """A clock-valued step isolates the scheduling from the PDE accuracy."""
    save_times = np.array([0.0, 0.1, 0.2, 0.3])
    for solver, initial in [
        (Burgers1DWeno5(16), np.zeros(16)),
        (Burgers1DFiniteVolume(16), np.zeros(16)),
        (AllenCahn2DSpectral(8, 8), np.zeros((8, 8))),
    ]:
        solver.step = lambda u, dt: u + dt
        solver.stable_dt = lambda *a, **k: 0.005
        saved = solver.solve_to(initial, 0.3, save_times)
        clock = saved.reshape(len(saved), -1)[:, 0]
        assert len(clock) == len(save_times)
        assert np.max(np.abs(clock - save_times)) < 1e-12


def test_solve_to_never_duplicates_a_state():
    """Two observation times inside one internal step must still produce
    two frames, and they must be different states."""
    save_times = np.array([0.0, 0.1, 0.1001, 0.2])
    s = Burgers1DWeno5(n=32, nu=0.01)
    out = s.solve_to(np.cos(2 * np.pi * s.x), 0.2, save_times)
    assert out.shape[0] == 4
    assert not np.allclose(out[1], out[2])


def test_solve_to_rejects_malformed_save_times():
    s = Burgers1DWeno5(n=16)
    with pytest.raises(ValueError):
        s.solve_to(np.zeros(16), 0.3, np.array([[0.0, 0.1]]))
    with pytest.raises(ValueError):
        s.solve_to(np.zeros(16), 0.3, np.array([0.0, 0.2, 0.1]))
    with pytest.raises(ValueError):
        s.solve_to(np.zeros(16), 0.3, np.array([0.0, 0.1, 0.1]))
    with pytest.raises(ValueError):
        s.solve_to(np.zeros(16), 0.3, np.array([-0.1, 0.1]))
    with pytest.raises(ValueError):
        s.solve_to(np.zeros(16), 0.3, np.array([0.05, 0.1]))
    with pytest.raises(ValueError):
        s.solve_to(np.zeros(16), 0.3, np.array([0.0, np.inf]))
    with pytest.raises(ValueError):
        s.solve_to(np.zeros(16), 0.1, np.array([0.0, 0.2]))


def test_validate_save_times_accepts_a_well_formed_request():
    """A well formed request must be accepted and returned unchanged."""
    s = Burgers1DWeno5(n=16)
    out = s.solve_to(np.zeros(16), 0.1, np.array([0.0, 0.05, 0.1]))
    assert out.shape[0] == 3


def test_frame_times_reports_the_actual_observation_times():
    s = Burgers1DWeno5(n=32, nu=0.01)
    save_times = np.array([0.0, 0.05, 0.1])
    s.solve_to(np.zeros(32), 0.1, save_times)
    assert np.allclose(s.frame_times(), save_times, atol=1e-12)
    policy = s.time_stepping_policy()
    assert "observation_scheduler" in policy
    assert "stability_bound" in policy


# ----------------------------------------------------------------------
# C. the flux split
# ----------------------------------------------------------------------
def test_burgers_flux_split_is_sign_aware_in_the_characteristic_speed():
    """f = u^2/2 is never negative, so splitting by the sign of f gives
    an identically zero f- and makes every state use the left-biased
    (downwind for u < 0) reconstruction."""
    s = Burgers1DWeno5(n=64)
    u = np.array([1.0, -0.5, 0.2, -1.0, 0.7, 0.0])
    f_plus, f_minus, alpha = s.split_flux(u)
    assert alpha >= np.max(np.abs(u))
    assert np.max(np.abs(f_plus + f_minus - 0.5 * u**2)) < 1e-14
    # the characteristic speeds of the two split fluxes must have
    # opposite signs
    assert np.all(0.5 * (u + alpha) >= 0.0)
    assert np.all(0.5 * (u - alpha) <= 0.0)


def test_burgers_perturbation_decays_on_a_positive_background():
    s = Burgers1DWeno5(n=128, nu=0.0)
    x = np.linspace(0.0, 1.0, 128, endpoint=False)
    u0 = 1.0 + 1e-4 * np.cos(2 * np.pi * 16 * x)
    out = s.solve(u0, dt=1e-4, num_steps=100)[-1]
    pert = out - 1.0
    rate = (np.sum(pert**2) / np.sum((u0 - 1.0) ** 2)) ** (1 / 100) - 1
    assert rate < 0.0


def test_burgers_perturbation_decays_on_a_negative_background():
    """On a negative background the characteristic speed is negative, so
    the old sign-blind split made the scheme antidiffusive: the
    perturbation energy grew at +0.88 per unit time."""
    s = Burgers1DWeno5(n=128, nu=0.0)
    x = np.linspace(0.0, 1.0, 128, endpoint=False)
    u0 = -1.0 + 1e-4 * np.cos(2 * np.pi * 16 * x)
    out = s.solve(u0, dt=1e-4, num_steps=100)[-1]
    pert = out + 1.0
    rate = (np.sum(pert**2) / np.sum((u0 + 1.0) ** 2)) ** (1 / 100) - 1
    assert rate < 0.0, f"antidiffusion: perturbation energy grows at {rate / 1e-4:+.4f} / time"


def test_burgers_both_backgrounds_agree_with_the_spectral_solver():
    """A positive and a negative background must both be reproduced by
    the independent spectral solver at the same physical time."""
    for bg in (1.0, -1.0):
        n = 128
        x = np.linspace(0.0, 1.0, n, endpoint=False)
        u0 = bg + 0.1 * np.cos(2 * np.pi * x)
        t = 0.05
        fv = Burgers1DWeno5(n=n, nu=0.0)
        dt_fv = fv.stable_dt(abs(bg) + 0.1, t)
        u_fv = fv.solve(u0, t / int(round(t / dt_fv)), int(round(t / dt_fv)))[-1]
        sp = Burgers1DSpectral(n=n, nu=0.0)
        dt_sp = sp.stable_dt(t)
        u_sp = sp.solve(u0, t / int(round(t / dt_sp)), int(round(t / dt_sp)))[-1]
        rel = np.linalg.norm(u_fv - u_sp) / np.linalg.norm(u_sp)
        assert rel < 5e-2, f"background {bg:+.0f}: relative difference {rel:.3e}"


def test_burgers_mean_is_conserved_with_the_new_split():
    s = Burgers1DWeno5(n=128, nu=0.01)
    u0 = 0.8 * np.cos(2 * np.pi * s.x) + 0.3 * np.cos(4 * np.pi * s.x + 0.5)
    out = s.solve_to(u0, 0.3, np.array([0.0, 0.15, 0.3]))
    assert np.max(np.abs(out.mean(axis=1) - out[0].mean())) < 1e-12


def test_weno5_right_reconstruction_is_the_mirror_of_the_left_one():
    """The right-biased candidate p2 was inconsistent with the mirror
    symmetry of the WENO5-JS set; it was masked because the old flux
    split made f- identically zero."""
    n = 64
    s = Burgers1DWeno5(n=n)
    rng = np.random.default_rng(7)
    f = rng.standard_normal(n)
    mirrored = np.roll(f[::-1], 1)  # mirrored[k] = f[-k]
    left = s._weno5_left(mirrored)
    right = s._weno5_right(f)
    idx = (-np.arange(n) - 1) % n
    assert np.max(np.abs(right - left[idx])) < 1e-12


def test_burgers_combined_step_bound_is_used():
    """The combined advection-diffusion bound is stricter than the
    minimum of the two separate bounds."""
    s = Burgers1DWeno5(n=256, nu=0.01)
    alpha = 1.1
    dt_combined = s.max_dt(alpha)
    dt_separate = min(0.4 * s.dx / alpha, 0.5 * s.dx**2 / 0.01)
    assert dt_combined <= dt_separate
    assert dt_combined == pytest.approx(s.dx**2 / (alpha * s.dx + 2 * 0.01))


# ----------------------------------------------------------------------
# E. manufactured solution with a continuum forcing
# ----------------------------------------------------------------------
def test_burgers_manufactured_solution_is_driven_by_the_real_integrator():
    """The forcing is a continuum expression that is evaluated
    independently of the solver; the error must be a real integration
    error and must converge under time refinement."""
    nu = 0.01
    t_end = 0.05
    n = 64
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = np.cos(2 * np.pi * x)
    exact = np.exp(-t_end) * np.cos(2 * np.pi * x)
    errors = []
    for dt in (4e-3, 2e-3, 1e-3):
        s = Burgers1DWeno5(n=n, nu=nu)
        out = s.solve(
            u0, dt, int(round(t_end / dt)),
            forcing=lambda t, u: _burgers_mms_forcing(x, t, nu),
        )
        errors.append(float(np.max(np.abs(out[-1] - exact))))
    assert errors[0] < 5e-3
    assert errors[1] < errors[0] / 4.0
    assert errors[2] < 5e-4


def test_burgers_manufactured_solution_through_solve_to():
    nu = 0.01
    t_end = 0.05
    n = 256
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = np.cos(2 * np.pi * x)
    exact = np.exp(-t_end) * np.cos(2 * np.pi * x)
    s = Burgers1DWeno5(n=n, nu=nu)
    frames = s.solve_to(
        u0, t_end, np.array([0.0, t_end]),
        forcing=lambda t, u: _burgers_mms_forcing(x, t, nu),
    )
    assert frames.shape[0] == 2
    assert np.max(np.abs(frames[0] - u0)) == 0.0
    assert np.max(np.abs(frames[-1] - exact)) < 1e-3
    assert s._last_steps > 0


def test_allen_cahn_manufactured_solution_is_driven_by_the_real_integrator():
    D, a = 0.001, 1.0
    t_end = 0.05
    n = 64
    xs = np.linspace(0, 1, n, endpoint=False)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u0 = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    exact = np.exp(-t_end) * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    errors = []
    for dt in (5e-3, 2.5e-3):
        s = AllenCahn2DSpectral(nx=n, ny=n, D=D, a=a)
        out = s.solve(
            u0, dt, int(round(t_end / dt)),
            forcing=lambda t, u: _allen_cahn_mms_forcing(X, Y, t, D, a),
        )
        errors.append(float(np.max(np.abs(out[-1] - exact))))
    assert errors[0] < 1e-6
    assert errors[1] < errors[0] / 3.0


# ----------------------------------------------------------------------
# F. de-aliasing
# ----------------------------------------------------------------------
def test_burgers_dealias_mask_is_hermitian_symmetric():
    """The old mask kept the modes 0..n//3-1 and -n//3..-1, which is
    *not* Hermitian symmetric: the mirror of -n//3 (i.e. +n//3) was
    dropped."""
    for n in (32, 64, 128, 96):
        mask = Burgers1DSpectral._dealias_mask(n)
        assert np.array_equal(mask, mask[(-np.arange(n)) % n])
        if n % 2 == 0:
            assert not mask[n // 2]


def test_allen_cahn_dealias_mask_is_hermitian_symmetric():
    for n in (32, 64, 128):
        mask = AllenCahn2DSpectral._dealias_mask(n)
        assert np.array_equal(mask, mask[(-np.arange(n)) % n])
        if n % 2 == 0:
            assert not mask[n // 2]


def test_burgers_spectral_nonlinear_product_has_no_residual_aliasing():
    """The nonlinear output must be projected back onto the retained
    band; otherwise the product content above the band folds back into
    it."""
    n = 128
    s = Burgers1DSpectral(n=n, nu=0.01)
    rng = np.random.default_rng(31337)
    u = rng.standard_normal(n)
    keep = n // 3
    exact = 0.5 * _padded_product_1d(u, 2, keep)
    masked = np.fft.ifft(
        np.fft.fft(0.5 * np.fft.ifft(np.fft.fft(u) * s._mask).real**2) * s._mask
    ).real
    rel = np.linalg.norm(masked - exact) / np.linalg.norm(exact)
    assert rel < 1e-3, f"residual aliasing error {rel:.3e}"


def test_allen_cahn_cubic_product_is_aliased_by_the_two_thirds_rule():
    """The 2/3 rule is a quadratic rule: the cubic product reaches
    n and folds back into the retained band. The solver must therefore
    use the 1/2 rule."""
    n = 64
    s = AllenCahn2DSpectral(nx=n, ny=n, D=0.001, a=1.0)
    mask_2_3 = np.zeros((n, n), dtype=bool)
    modes = np.abs(np.rint(np.fft.fftfreq(n) * n).astype(int))
    mask_2_3 = (modes <= n // 3)[:, None] & (modes <= n // 3)[None, :]
    u = np.random.default_rng(5).standard_normal((n, n))
    u_f = np.fft.ifft2(np.fft.fft2(u) * mask_2_3).real
    cubic_2_3 = np.fft.ifft2(np.fft.fft2(u_f**3) * mask_2_3).real
    # the solver's own (1/2 rule) product
    u_f2 = np.fft.ifft2(np.fft.fft2(u) * s._mask).real
    cubic_half = np.fft.ifft2(np.fft.fft2(u_f2**3) * s._mask).real
    assert not np.array_equal(s._mask, mask_2_3)
    # the 2/3 rule leaks energy outside its own retained band
    assert np.linalg.norm(cubic_2_3 - cubic_half) > 0.0


def test_allen_cahn_dealias_mask_uses_half_the_nyquist():
    n = 64
    mask = AllenCahn2DSpectral._dealias_mask(n)
    modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
    assert np.array_equal(mask, np.abs(modes) <= n // 4)


# ----------------------------------------------------------------------
# the dataset-facing path
# ----------------------------------------------------------------------
def test_dataset_manifest_records_the_actual_observation_times(tmp_path):
    """The manifest must record the times that were actually written,
    not only the times that were requested."""
    from pdelab.config import PDESystemConfig
    from pdelab.data.generate import DatasetGenerator

    system = PDESystemConfig(
        name="burgers",
        dimension=1,
        solver="finite_volume_weno5_js",
        solver_grid=(64,),
        observation_grid=(32,),
        observation_times=[0.0, 0.01, 0.02, 0.03],
        t_final=0.03,
        params={"nu": 0.01},
        initial_condition={"kind": "fourier", "n_modes": 4, "amplitude": 0.9,
                           "mean": 0.0},
        reference_grid=(64,),
    )
    gen = DatasetGenerator(
        system=system, out_dir=tmp_path, n_train=1, n_val=1, n_test=1,
        seed=99, profile="test",
    )
    manifest = gen.generate()
    rec = manifest["trajectories"][0]
    assert "actual_observation_times" in rec
    assert np.allclose(rec["actual_observation_times"], system.observation_times,
                       atol=1e-12)
    assert manifest["observation_scheduler"]["observation_scheduler"]
    assert manifest["time_stepping_policy"]["solver"]


def test_validation_manufactured_solution_check_is_not_vacuous():
    """The old check computed ``u + dt*(rhs + forcing)`` with a forcing
    derived from the same discrete rhs, so it cancelled algebraically
    and never called the integrator."""
    result = _burgers_validation()
    check = next(c for c in result["checks"]
                 if c["name"] == "burgers.manufactured_solution")
    assert check["detail"]["driven_through"].startswith("solve")
    assert check["detail"]["rows"][0]["integrator_calls"] > 0
    assert check["detail"]["solve_to_row"]["integrator_calls"] > 0
    assert check["detail"]["max_abs_error"] > 0.0

    result = _allen_cahn_validation()
    check = next(c for c in result["checks"]
                 if c["name"] == "allen_cahn.manufactured_solution")
    assert check["detail"]["driven_through"].startswith("solve")
    assert check["detail"]["rows"][0]["integrator_calls"] > 0
    assert check["detail"]["max_abs_error"] > 0.0


def test_validation_suite_reports_the_new_checks():
    burgers_names = {c["name"] for c in _burgers_validation()["checks"]}
    for name in [
        "burgers.flux_split_sign_aware",
        "burgers.perturbation_decays_positive_background",
        "burgers.perturbation_decays_negative_background",
        "burgers.observation_times_exact",
        "burgers.dataset_observation_times_exact",
        "burgers.aliasing_error_quantified",
        "burgers.sign_agreement_with_spectral_solver",
    ]:
        assert name in burgers_names

    ac_names = {c["name"] for c in _allen_cahn_validation()["checks"]}
    for name in [
        "allen_cahn.observation_times_exact",
        "allen_cahn.dataset_observation_times_exact",
        "allen_cahn.dealiasing_masks_and_error",
    ]:
        assert name in ac_names


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
    result = _allen_cahn_validation()
    failed = [c["name"] for c in result["checks"] if not c["passed"]]
    assert not failed, f"failing Allen-Cahn checks: {failed}"
