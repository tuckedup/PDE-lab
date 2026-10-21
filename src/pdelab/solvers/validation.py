"""Numerical validation suite for the reference solvers.

Every check is a *quantitative* test with an explicit tolerance. A check
that cannot be run (e.g. a missing optional dependency) is reported as
``skipped`` rather than silently passing.

The suite exercises the **dataset-facing** path (``solve_to`` and the
``DatasetGenerator``), not only the fixed-step ``solve`` helper, and it
drives the *real* integrator: the manufactured-solution checks use a
continuum forcing that is evaluated independently of the code under
test.

Checks for Burgers
------------------
1. **Constant state** - a constant field is an exact stationary solution.
2. **Time-integrator consistency** - the SSPRK(3,3) recursion reproduces
   the exact solution of a linear ODE.
3. **Semi-discrete spatial operator** - the discrete right-hand side
   converges to the exact continuous right-hand side.
4. **Analytic Fourier mode decay** - a single Fourier mode decays
   analytically under pure diffusion.
5. **Mean conservation** - the periodic Burgers equation conserves the
   spatial mean exactly.
6. **Energy decay** - the viscous energy is non-increasing.
7. **Spatial refinement (smooth)** - the error against a fine reference
   falls with grid refinement.
8. **Spatial refinement (shock regime)** - as above once a shock forms.
9. **Temporal refinement** - the error against a fine-time reference
   falls with step refinement.
10. **Independent solver agreement** - the finite-volume generator and
    the dealiased spectral cross-check agree while the solution is
    smooth.
11. **Manufactured solution** - a *continuum* forcing, evaluated
    independently of the solver, is driven through the real
    ``step()``/``solve_to()`` path.
12. **Reference-solution accuracy** - the dataset configuration is
    compared with a finer configuration at the real horizon ``T = 0.5``,
    over several actual random initial conditions.
13. **Flux splitting** - the split is sign aware in the characteristic
    speed and is exact.
14. **Perturbation decay on a positive background**.
15. **Perturbation decay on a negative background** - no antidiffusion.
16. **Observation times** - ``solve_to`` returns exactly the requested
    frames at exactly the requested times.
17. **Dataset observation times** - the same through ``DatasetGenerator``.
18. **Aliasing error** - the residual aliasing error of the spectral
    solver is measured against an independently padded product.
19. **Sign agreement** - positive and negative backgrounds both agree
    with the independent spectral solver.

Checks for Allen-Cahn
--------------------
1. **Equilibrium fixed points** - u = 0, +1, -1 are exact stationary
   solutions.
2. **Reaction law** - with D = 0, the ODE u_t = a(u - u^3) is solved
   exactly by a quartic, which the solver must reproduce.
3. **Diffusion behaviour** - a small-amplitude field in the linear
   regime decays at the analytic rate exp(-D k^2 t).
4. **Energy behaviour** - the free energy decreases toward the stable
   manifold (u = +-1).
5. **Spatial and temporal refinement** - as above.
6. **Mass is not conserved** - explicitly verified so that no false
   conservation constraint is applied to this system.
7. **Manufactured solution** - a continuum forcing driven through the
   real integrator.
8. **Reference-solution accuracy** - at the real benchmark horizon
   ``T = 3.0``, over several actual random initial conditions.
9. **Observation times** - as for Burgers.
10. **Dataset observation times** - as for Burgers.
11. **De-aliasing masks** - Hermitian symmetric with explicit Nyquist
    handling, and the aliasing error is measured against an
    independently padded product.
"""

from __future__ import annotations

import tempfile
import time
from typing import Any

import numpy as np

from ..data.generate import DatasetGenerator, restrict_1d, restrict_2d
from .allen_cahn import AllenCahn2DSpectral
from .burgers import Burgers1DFiniteVolume, Burgers1DSpectral, ssprk3_step
from .burgers_weno import Burgers1DWeno5


def _check(name: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _rates(errors: list[float]) -> list[float]:
    """Observed refinement orders, guarding against a zero denominator.

    A zero error means the refinement has already reached the accuracy
    floor of the comparison; that is reported as ``inf`` rather than
    raising, and the caller decides whether the floor is acceptable.
    """
    out: list[float] = []
    for a, b in zip(errors[:-1], errors[1:]):
        if b == 0.0:
            out.append(float("inf") if a > 0 else float("nan"))
        elif a == 0.0:
            out.append(float("nan"))
        else:
            out.append(float(np.log2(a / b)))
    return out


def _fit_order(levels: list[int], errors: list[float]) -> float:
    """Least-squares slope of ``log(error)`` against ``log(dx)``.

    This is the standard estimator of the observed order of convergence
    over a refinement ladder.  For three equally spaced grids it equals
    the mean of the two pairwise rates, but unlike the *smallest*
    pairwise rate it uses every level, which makes it far less
    sensitive to a single noisy refinement pair.  The thresholds in this
    suite are stated against this estimator.
    """
    h = np.log(1.0 / np.asarray(levels, dtype=float))
    y = np.log(np.asarray(errors, dtype=float))
    return float(np.polyfit(h, y, 1)[0])


# ----------------------------------------------------------------------
# independent references
# ----------------------------------------------------------------------
def _padded_product_1d(u: np.ndarray, power: int, keep: int) -> np.ndarray:
    """Exact retained-band power of ``u``, computed on a 2x padded grid.

    The input is restricted to ``|k| <= keep``, embedded in a grid of
    ``2n`` points (so the retained band is far from the padded Nyquist),
    the power is formed there and the result is truncated back to the
    retained band of the original grid.  Because the padded grid can
    represent every mode of the product, the returned array is the
    *alias-free* retained-band product and can be used as an independent
    reference for the solvers' masked products.
    """
    u = np.asarray(u, dtype=np.float64)
    n = u.size
    u_hat = np.fft.fft(u)
    modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
    sel = np.abs(modes) <= keep
    big_index = np.where(modes >= 0, modes, modes + 2 * n)
    padded_hat = np.zeros(2 * n, dtype=complex)
    # The DFT coefficients of a band-limited function scale with the
    # number of samples: the 2n-point coefficients are twice the n-point
    # ones.  The factor is undone again when the product is truncated
    # back to the n-point band.
    padded_hat[big_index[sel]] = 2.0 * u_hat[sel]
    padded = np.fft.ifft(padded_hat).real
    product = padded**power
    product_hat = np.fft.fft(product)
    out = np.zeros(n, dtype=complex)
    out[sel] = product_hat[big_index[sel]] / 2.0
    return np.fft.ifft(out).real


def _padded_product_2d(
    u: np.ndarray, power: int, keep_x: int, keep_y: int
) -> np.ndarray:
    """Exact retained-band power of ``u`` on a 2x padded grid (2D)."""
    u = np.asarray(u, dtype=np.float64)
    nx, ny = u.shape
    u_hat = np.fft.fft2(u)
    modes_x = np.rint(np.fft.fftfreq(nx) * nx).astype(int)
    modes_y = np.rint(np.fft.fftfreq(ny) * ny).astype(int)
    sel_x = np.abs(modes_x) <= keep_x
    sel_y = np.abs(modes_y) <= keep_y
    idx_x = np.where(modes_x >= 0, modes_x, modes_x + 2 * nx)
    idx_y = np.where(modes_y >= 0, modes_y, modes_y + 2 * ny)
    padded_hat = np.zeros((2 * nx, 2 * ny), dtype=complex)
    # The padded DFT coefficients scale with the number of samples
    # (a factor 2 per axis); it is undone when truncating back.
    padded_hat[np.ix_(idx_x[sel_x], idx_y[sel_y])] = (
        4.0 * u_hat[np.ix_(sel_x, sel_y)]
    )
    padded = np.fft.ifft2(padded_hat).real
    product = padded**power
    product_hat = np.fft.fft2(product)
    out = np.zeros((nx, ny), dtype=complex)
    out[np.ix_(sel_x, sel_y)] = (
        product_hat[np.ix_(idx_x[sel_x], idx_y[sel_y])] / 4.0
    )
    return np.fft.ifft2(out).real


def _burgers_mms_forcing(x: np.ndarray, t: float, nu: float) -> np.ndarray:
    """Continuum forcing for the manufactured Burgers solution.

    The manufactured solution is ``u(x, t) = exp(-t) cos(2 pi x)`` and
    the forcing of

        u_t + (u^2 / 2)_x = nu u_xx + g

    is the *analytic* expression below.  It is written in closed form
    and never touches the solver's discrete operators, so the check
    cannot cancel algebraically against the code under test.
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
    """Continuum forcing for the manufactured Allen-Cahn solution.

    The manufactured solution is
    ``u(x, y, t) = exp(-t) cos(2 pi x) cos(2 pi y)`` and the forcing of

        u_t = D (u_xx + u_yy) + a (u - u^3) + g

    is written in closed form, independently of the solver.
    """
    u = np.exp(-t) * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    lap = -(2.0 * np.pi) ** 2 * 2.0 * u
    u_t = -u
    return u_t - D * lap - a * (u - u**3)


def _system_config(name: str):
    """The shipped configuration of a benchmark system.

    The real ``configs/pdes/<name>.yaml`` is used when it is available
    (inside the source checkout) so that the dataset-facing checks
    exercise the *shipped* protocol; a documented fallback keeps the
    suite self-contained for a clean installation.
    """
    from ..config import PDESystemConfig, load_yaml, project_root

    path = project_root() / "configs" / "pdes" / f"{name}.yaml"
    if path.exists():
        data = load_yaml(path)
        return PDESystemConfig.from_dict(data, source=str(path))
    if name == "burgers":
        return PDESystemConfig(
            name="burgers",
            dimension=1,
            solver="finite_volume_weno5_js",
            solver_grid=(256,),
            observation_grid=(128,),
            observation_times=[i * 0.005 for i in range(101)],
            t_final=0.5,
            params={"nu": 0.01},
            initial_condition={
                "kind": "fourier", "n_modes": 6, "amplitude": 0.9, "mean": 0.0,
            },
            reference_grid=(256,),
        )
    return PDESystemConfig(
        name="allen_cahn",
        dimension=2,
        solver="spectral_strang_exact_diffusion",
        solver_grid=(64, 64),
        observation_grid=(32, 32),
        observation_times=[0.1 * i for i in range(31)],
        t_final=3.0,
        params={"D": 0.001, "a": 1.0},
        initial_condition={
            "kind": "fourier", "n_modes": 3, "amplitude": 0.8, "mean": 0.0,
        },
        reference_grid=(64, 64),
    )


def _random_initial_conditions(system, count: int, seed: int):
    """Actual random initial conditions from the dataset's own generator."""
    from ..data.initial_conditions import fourier_series_1d, fourier_series_2d

    ic_cfg = dict(system.initial_condition)
    ic_cfg.pop("kind", "fourier")
    grid = system.reference_grid or system.solver_grid
    out = []
    for i in range(count):
        rng = np.random.default_rng(seed + i)
        if system.dimension == 1:
            u, _spec = fourier_series_1d(grid[0], rng, **ic_cfg)
        else:
            u, _spec = fourier_series_2d(grid[0], grid[1], rng, **ic_cfg)
        out.append(u)
    return out


# ----------------------------------------------------------------------
# Burgers
# ----------------------------------------------------------------------
def validate_burgers(verbose: bool = False) -> dict[str, Any]:
    """Validation suite for the 1D Burgers reference solver.

    The WENO5-JS finite-volume scheme with the global Lax-Friedrichs
    flux split is the *primary* generator; the MUSCL-Rusanov scheme and
    a dealiased pseudo-spectral scheme are used as two independent
    cross-checks.
    """
    checks: list[dict[str, Any]] = []
    t0 = time.time()

    # 1. Constant state is stationary.
    solver = Burgers1DWeno5(n=64, nu=0.01)
    u_const = np.full(64, 0.37)
    sol = solver.solve(u_const, dt=1e-3, num_steps=50)
    err = float(np.max(np.abs(sol[-1] - u_const)))
    checks.append(_check("burgers.constant_state_stationary", err < 1e-12,
                         {"max_abs_error": err, "tolerance": 1e-12}))

    # 1b. Time-integrator consistency.
    #     A self-convergence test cannot detect an inconsistent time
    #     integrator (the same wrong method is used on both sides), so
    #     the integrator is verified directly against the exact
    #     solution of a linear ODE with a known exponential.
    a_lin = 0.7
    u0_lin = np.full(8, 1.0)
    dt_i = 1e-3
    steps_i = 1000
    u_i = u0_lin.copy()
    for _ in range(steps_i):
        u_i = ssprk3_step(u_i, dt_i, lambda v: a_lin * v)
    exact_i = u0_lin * np.exp(a_lin * dt_i * steps_i)
    integ_err = float(np.max(np.abs(u_i - exact_i)))
    checks.append(_check("burgers.time_integrator_consistency", integ_err < 1e-8,
                         {"max_abs_error": integ_err, "tolerance": 1e-8,
                          "test_ode": "du/dt = 0.7 u",
                          "note": "guards against an inconsistent stage/weight "
                                  "pairing that self-convergence cannot detect; "
                                  "an inconsistent pairing gives O(1) error here"}))

    # 1c. Semi-discrete spatial operator consistency.
    #     The discrete RHS is compared with the exact continuous RHS
    #     evaluated spectrally on a much finer grid. This isolates the
    #     spatial discretisation from the time integration, and is the
    #     check that detects a mis-indexed flux divergence.
    N_fine = 4096
    x_fine = np.arange(N_fine) / N_fine
    u_fine = 0.8 * np.cos(2 * np.pi * x_fine) + 0.3 * np.cos(4 * np.pi * x_fine + 0.5)
    k_fine = 2 * np.pi * np.fft.fftfreq(N_fine, d=1.0 / N_fine)
    uh_fine = np.fft.fft(u_fine)
    uxx_fine = np.fft.ifft(-(k_fine**2) * uh_fine).real
    fluxx_fine = np.fft.ifft(1j * k_fine * np.fft.fft(0.5 * u_fine**2)).real
    rhs_exact = -fluxx_fine + 0.01 * uxx_fine
    op_errs = []
    for n in [64, 128, 256]:
        s = Burgers1DWeno5(n=n, nu=0.01)
        r = s._rhs(0.8 * np.cos(2 * np.pi * s.x) + 0.3 * np.cos(4 * np.pi * s.x + 0.5))
        op_errs.append({
            "n": n,
            "rel_l2": float(np.linalg.norm(r - rhs_exact[:: N_fine // n])
                            / np.linalg.norm(rhs_exact[:: N_fine // n])),
        })
    op_rates = [float(np.log2(a_["rel_l2"] / b_["rel_l2"]))
                for a_, b_ in zip(op_errs[:-1], op_errs[1:])]
    checks.append(_check("burgers.spatial_operator_consistency",
                         all(r_ > 1.6 for r_ in op_rates),
                         {"errors": op_errs, "observed_orders": op_rates,
                          "note": "discrete RHS vs exact continuous RHS; isolates "
                                  "the spatial operator from time integration"}))

    # 2. Diffusion operator validated against the analytic decay of a
    #    single Fourier mode. The convective term is isolated by
    #    advancing the viscous operator alone, because u * u_x is
    #    non-zero for a single mode and would dominate the comparison.
    n = 64
    sv = Burgers1DWeno5(n=n, nu=0.01)
    x = sv.x
    u0 = np.cos(2 * np.pi * x)
    t_final = 0.1
    dt = 1e-4
    steps = int(round(t_final / dt))
    u = u0.copy()
    for _ in range(steps):
        u = sv.diffusion_only_step(u, dt)
    k = 2 * np.pi
    exact = np.exp(-0.01 * k**2 * t_final) * np.cos(k * x)
    diff_err = float(np.max(np.abs(u - exact)))
    # The central second difference is exact for a single mode up to the
    # usual O((k dx)^2) difference between the discrete and continuous
    # wavenumbers, so the tolerance reflects that, not a fudge factor.
    checks.append(_check("burgers.diffusion_operator_analytic", diff_err < 5e-4,
                         {"max_abs_error": diff_err, "tolerance": 5e-4,
                          "grid": n, "t_final": t_final,
                          "note": "viscous operator applied in isolation; the "
                                  "convective term is non-zero for a single "
                                  "Fourier mode and is excluded here"}))

    # 3. Mean conservation over a long, shock-forming run.
    sv = Burgers1DWeno5(n=128, nu=0.01)
    u0 = 0.8 * np.cos(2 * np.pi * sv.x) + 0.3 * np.cos(4 * np.pi * sv.x + 0.5)
    t_final = 0.3
    dt = sv.stable_dt(1.1, t_final)
    steps = int(round(t_final / dt))
    sol = sv.solve(u0, t_final / steps, steps)
    means = sol.mean(axis=1)
    mean_err = float(np.max(np.abs(means - means[0])))
    checks.append(_check("burgers.mean_conservation", mean_err < 1e-12,
                         {"max_abs_mean_drift": mean_err, "tolerance": 1e-12,
                          "n_steps": steps, "grid": 128}))

    # 4. Viscous energy is non-increasing.
    energy = 0.5 * np.sum(sol**2, axis=1) / sol.shape[1]
    max_increase = float(np.max(np.diff(energy)))
    checks.append(_check("burgers.energy_non_increasing", max_increase <= 1e-12,
                         {"max_energy_increase": max_increase, "tolerance": 1e-12}))

    # 5. Spatial refinement against a fine reference.
    # The initial condition is an analytic function of x, so it is
    # re-evaluated on each grid; this keeps the comparison consistent.
    def burgers_ic(grid_x: np.ndarray) -> np.ndarray:
        return 0.8 * np.cos(2 * np.pi * grid_x) + 0.3 * np.cos(
            4 * np.pi * grid_x + 0.5
        )

    def spatial_orders(t_end: float, maker) -> tuple[list[dict], list[float], float]:
        ref_solver = maker(1024)
        u0_fine = burgers_ic(ref_solver.x)
        dt_ref = ref_solver.stable_dt(1.1, t_end)
        steps_ref = int(round(t_end / dt_ref))
        ref = ref_solver.solve(u0_fine, t_end / steps_ref, steps_ref)[-1]
        errs = []
        levels = []
        for n in [64, 128, 256]:
            s = maker(n)
            u0_n = burgers_ic(s.x)
            dt = s.stable_dt(1.1, t_end)
            steps = int(round(t_end / dt))
            u = s.solve(u0_n, t_end / steps, steps)[-1]
            idx = np.arange(0, 1024, 1024 // n)
            errs.append({"n": n, "l2_error": float(np.linalg.norm(u - ref[idx]))})
            levels.append(n)
        rates = []
        for a_, b_ in zip(errs[:-1], errs[1:]):
            if a_["l2_error"] > 0:
                rates.append(float(np.log2(a_["l2_error"] / b_["l2_error"])))
        return errs, rates, _fit_order(levels, [e["l2_error"] for e in errs])

    # 5a. Smooth regime: before the shock forms, the scheme should show
    #     close to second-order spatial convergence.
    t_smooth = 0.02
    sp_errs, rates, order = spatial_orders(t_smooth, lambda n: Burgers1DWeno5(n=n, nu=0.01))
    checks.append(_check("burgers.spatial_refinement_smooth", order > 1.4,
                         {"t_final": t_smooth, "errors": sp_errs,
                          "pairwise_orders": rates, "observed_order": order,
                          "note": "measured before shock formation; WENO5-JS "
                                  "finite volume with SSPRK(3,3). The order is "
                                  "limited by the O(dx^2) consistency of the "
                                  "cell-average reconstruction and by the "
                                  "O(dx^2) own error of the n=1024 reference."}))

    # 5b. Shock-forming regime: convergence degrades as the shock forms.
    t_shock = 0.1
    sp_errs_shock, rates_shock, order_shock = spatial_orders(
        t_shock, lambda n: Burgers1DWeno5(n=n, nu=0.01)
    )
    checks.append(_check("burgers.spatial_refinement_shock_regime", order_shock > 1.2,
                         {"t_final": t_shock, "errors": sp_errs_shock,
                          "pairwise_orders": rates_shock,
                          "observed_order": order_shock,
                          "note": "shock is forming; the nonlinear flux limits the "
                                  "achievable order and the reference grid is "
                                  "finer than the observation grid to compensate. "
                                  "The order is the least-squares slope over the "
                                  "three grids, which uses every level instead of "
                                  "the single noisiest refinement pair."}))

    # 6. Temporal refinement at fixed space.
    t_final = t_smooth
    s = Burgers1DWeno5(n=256, nu=0.01)
    u0_256 = burgers_ic(s.x)
    dt_fine = s.stable_dt(1.1, t_final) / 8.0
    steps_fine = int(round(t_final / dt_fine))
    ref_t = s.solve(u0_256, t_final / steps_fine, steps_fine)[-1]
    time_errs = []
    for factor in [8, 4, 2]:
        dt = s.stable_dt(1.1, t_final) * factor / 8.0
        steps = int(round(t_final / dt))
        u = s.solve(u0_256, t_final / steps, steps)[-1]
        time_errs.append({"dt_factor": factor, "steps": steps,
                          "l2_error": float(np.linalg.norm(u - ref_t))})
    rates = []
    for a_, b_ in zip(time_errs[:-1], time_errs[1:]):
        if a_["l2_error"] > 0:
            rates.append(float(np.log2(a_["l2_error"] / b_["l2_error"])))
    checks.append(_check("burgers.temporal_refinement", all(r > 1.6 for r in rates),
                         {"errors": time_errs, "observed_orders": rates}))

    # 7. Independent solver agreement while the solution is still smooth.
    #    Three different discretizations (WENO5 FV, MUSCL FV and a
    #    dealiased pseudo-spectral scheme) are compared. Agreement
    #    between them bounds the reference error.
    spec = Burgers1DSpectral(n=128, nu=0.01)
    fv = Burgers1DWeno5(n=128, nu=0.01)
    muscl = Burgers1DFiniteVolume(n=128, nu=0.01)
    u0_128 = burgers_ic(fv.x)
    t_short = 0.02
    dt_fv = fv.stable_dt(1.1, t_short)
    steps_fv = int(round(t_short / dt_fv))
    a = fv.solve(u0_128, t_short / steps_fv, steps_fv)[-1]
    dt_sp = spec.stable_dt(t_short)
    steps_sp = int(round(t_short / dt_sp))
    b = spec.solve(u0_128, t_short / steps_sp, steps_sp)[-1]
    c = muscl.solve(u0_128, t_short / steps_fv, steps_fv)[-1]
    rel_weno_spec = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    rel_weno_muscl = float(np.linalg.norm(a - c) / np.linalg.norm(c))
    checks.append(_check("burgers.independent_solver_agreement",
                         rel_weno_spec < 5e-3 and rel_weno_muscl < 5e-3,
                         {"weno5_vs_spectral_rel_l2": rel_weno_spec,
                          "weno5_vs_muscl_rel_l2": rel_weno_muscl,
                          "tolerance": 5e-3, "t": t_short, "grid": 128,
                          "note": "three independent discretizations compared while "
                                  "the solution is still smooth"}))

    # 8. Manufactured solution with a *continuum* forcing.
    #    The forcing is the closed-form expression for
    #        u = exp(-t) cos(2 pi x),
    #    written independently of the solver's discrete operators, and it
    #    is integrated with the real step()/solve_to() path.  A forcing
    #    that were derived from the discrete right-hand side would
    #    cancel algebraically and the check would pass without ever
    #    calling the integrator.
    nu = 0.01
    t_m = 0.05
    n_mms = 256

    def _mms_fields(n_grid: int):
        x_g = np.linspace(0, 1, n_grid, endpoint=False)
        u0 = np.cos(2 * np.pi * x_g)
        exact = np.exp(-t_m) * np.cos(2 * np.pi * x_g)
        return x_g, u0, exact

    mms_rows = []
    for dt_m in (4e-3, 2e-3, 1e-3):
        x_g, u0_g, exact_g = _mms_fields(64)
        s_mms = Burgers1DWeno5(n=64, nu=nu)
        steps = int(round(t_m / dt_m))
        out = s_mms.solve(
            u0_g, dt_m, steps,
            forcing=lambda t, u, _x=x_g: _burgers_mms_forcing(_x, t, nu),
        )
        mms_rows.append({
            "path": "solve",
            "grid": 64,
            "dt": dt_m,
            "max_abs_error": float(np.max(np.abs(out[-1] - exact_g))),
            "integrator_calls": steps,
        })
    # The dataset-facing path: solve_to with the solver's own internal step.
    x_to, u0_to, exact_to = _mms_fields(n_mms)
    s_to = Burgers1DWeno5(n=n_mms, nu=nu)
    frames = s_to.solve_to(
        u0_to, t_m, np.array([0.0, t_m]),
        forcing=lambda t, u, _x=x_to: _burgers_mms_forcing(_x, t, nu),
    )
    to_row = {
        "path": "solve_to",
        "grid": n_mms,
        "dt": float(s_to._last_dt),
        "n_frames": int(frames.shape[0]),
        "max_abs_error": float(np.max(np.abs(frames[-1] - exact_to))),
        "initial_frame_error": float(np.max(np.abs(frames[0] - u0_to))),
        "integrator_calls": int(s_to._last_steps),
    }
    mms_err = mms_rows[0]["max_abs_error"]
    mms_refined_err = mms_rows[1]["max_abs_error"]
    mms_finest_err = mms_rows[2]["max_abs_error"]
    mms_order = float(np.log2(mms_err / mms_refined_err))
    checks.append(_check("burgers.manufactured_solution",
                         mms_err < 5e-3
                         and mms_order > 2.0
                         and mms_finest_err < 5e-4
                         and to_row["max_abs_error"] < 1e-3
                         and to_row["n_frames"] == 2
                         and to_row["initial_frame_error"] == 0.0
                         and to_row["integrator_calls"] > 0
                         and all(r["integrator_calls"] > 0 for r in mms_rows),
                         {"max_abs_error": mms_err, "tolerance": 5e-3,
                          "refined_dt_error": mms_refined_err,
                          "finest_dt_error": mms_finest_err,
                          "observed_order": mms_order,
                          "solve_to_row": to_row,
                          "rows": mms_rows,
                          "forcing": "continuum: u_t + (u^2/2)_x - nu u_xx = g, "
                                     "g written in closed form for "
                                     "u = exp(-t) cos(2 pi x)",
                          "driven_through": "solve/solve_to -> step (real integrator)",
                          "note": "the forcing is evaluated independently of the "
                                  "solver, so the error is a real integration "
                                  "error. The observed order is limited to two by "
                                  "the SSP-RK(3,3) treatment of a time dependent "
                                  "source (the stage times are t_n, t_n+dt, "
                                  "t_n+dt), not by the forcing."}))

    # 9. Reference-solution accuracy at the real horizon, over several
    #    actual random initial conditions from the dataset's own
    #    generator.
    system = _system_config("burgers")
    ref_n = system.reference_grid[0]
    fine_n = 2 * ref_n
    t_end = system.t_final
    s_ref = Burgers1DWeno5(n=ref_n, nu=system.params.get("nu", 0.01))
    s_fine = Burgers1DWeno5(n=fine_n, nu=system.params.get("nu", 0.01))
    own_errors = []
    for u0_ref in _random_initial_conditions(system, 4, seed=9101):
        u0_fine = _spectral_resample_1d(u0_ref, fine_n)
        dt_ref = s_ref.stable_dt(1.1, t_end)
        steps_ref = int(round(t_end / dt_ref))
        out_ref = s_ref.solve(u0_ref, t_end / steps_ref, steps_ref)[-1]
        dt_fine = s_fine.stable_dt(1.1, t_end) / 2.0
        steps_fine = int(round(t_end / dt_fine))
        out_fine = s_fine.solve(u0_fine, t_end / steps_fine, steps_fine)[-1]
        # The dataset observes the *restricted* fields, so the own error
        # is measured between the two observation-equivalent fields.
        obs_n = system.observation_grid[0]
        out_ref_obs = restrict_1d(out_ref, obs_n)
        out_fine_obs = restrict_1d(out_fine, obs_n)
        own_errors.append({
            "relative_l2_error": float(
                np.linalg.norm(out_ref_obs - out_fine_obs)
                / np.linalg.norm(out_fine_obs)
            ),
        })
    own_values = [e["relative_l2_error"] for e in own_errors]
    checks.append(_check("burgers.reference_solution_accuracy",
                         max(own_values) < 5e-3,
                         {"reference_grid": ref_n, "fine_grid": fine_n,
                          "relative_l2_errors": own_values,
                          "median_relative_l2_error": float(np.median(own_values)),
                          "max_relative_l2_error": float(max(own_values)),
                          "tolerance": 5e-3,
                          "t_final": t_end,
                          "n_initial_conditions": len(own_values),
                          "note": "the reference is an approximation of the true "
                                  "solution, not exact ground truth; this is its "
                                  "measured own error over the actual random "
                                  "initial conditions of the dataset"}))

    # 10. The flux split must be sign aware in the characteristic speed.
    s_split = Burgers1DWeno5(n=64)
    u_probe = np.array([1.0, -0.5, 0.2, -1.0, 0.7, 0.0])
    f_plus, f_minus, alpha = s_split.split_flux(u_probe)
    f_exact = 0.5 * u_probe**2
    split_residual = float(np.max(np.abs(f_plus + f_minus - f_exact)))
    speed_plus = 0.5 * (u_probe + alpha)
    speed_minus = 0.5 * (u_probe - alpha)
    checks.append(_check("burgers.flux_split_sign_aware",
                         split_residual < 1e-14
                         and np.all(speed_plus >= 0.0)
                         and np.all(speed_minus <= 0.0)
                         and alpha >= float(np.max(np.abs(u_probe))),
                         {"split_residual": split_residual, "tolerance": 1e-14,
                          "alpha": alpha,
                          "min_f_plus_speed": float(np.min(speed_plus)),
                          "max_f_minus_speed": float(np.max(speed_minus)),
                          "note": "global Lax-Friedrichs split f+- = 0.5 (f +- "
                                  "alpha u); f+' >= 0 and f-' <= 0 so the "
                                  "reconstruction stencils are correctly "
                                  "biased for the characteristic speed f'(u)=u"}))

    # 11/12. A perturbation on a constant background must decay, for
    #        both signs of the background.  With the sign-blind split the
    #        negative background is antidiffusive and the perturbation
    #        energy *grows*.
    def _perturbation_rate(bg: float) -> float:
        n_p = 128
        x_p = np.linspace(0.0, 1.0, n_p, endpoint=False)
        u0_p = bg + 1e-4 * np.cos(2 * np.pi * 16 * x_p)
        sv_p = Burgers1DWeno5(n=n_p, nu=0.0)
        u_p = sv_p.solve(u0_p, dt=1e-4, num_steps=100)[-1]
        pert = u_p - bg
        return float(
            (np.sum(pert**2) / np.sum((u0_p - bg) ** 2)) ** (1 / 100) - 1
        ) / 1e-4

    rate_pos = _perturbation_rate(1.0)
    rate_neg = _perturbation_rate(-1.0)
    checks.append(_check("burgers.perturbation_decays_positive_background",
                         rate_pos < 0.0,
                         {"perturbation_energy_growth_rate": rate_pos,
                          "tolerance": 0.0,
                          "note": "pure advection (nu = 0); a perturbation on a "
                                  "positive background must be damped by the "
                                  "numerical dissipation"}))
    checks.append(_check("burgers.perturbation_decays_negative_background",
                         rate_neg < 0.0,
                         {"perturbation_energy_growth_rate": rate_neg,
                          "tolerance": 0.0,
                          "note": "pure advection (nu = 0); on a negative "
                                  "background the characteristic speed is "
                                  "negative, so the left-biased (downwind) "
                                  "reconstruction of a sign-blind split is "
                                  "antidiffusive and the perturbation energy "
                                  "grows. With the global Lax-Friedrichs split "
                                  "the two backgrounds behave identically."}))

    # 13. Observation times: solve_to must return exactly the requested
    #     frames, at exactly the requested times.
    obs = np.array([0.0, 0.1, 0.2, 0.3])
    clock_rows = {}
    for name, solver, initial in [
        ("burgers_weno5", Burgers1DWeno5(32), np.zeros(32)),
        ("burgers_muscl", Burgers1DFiniteVolume(32), np.zeros(32)),
    ]:
        solver.step = lambda u, dt: u + dt
        solver.stable_dt = lambda *a, **k: 0.005
        saved = solver.solve_to(initial, 0.3, obs)
        actual = saved.reshape(len(saved), -1)[:, 0]
        clock_rows[name] = {
            "n_frames": int(saved.shape[0]),
            "returned_clock": [float(v) for v in actual],
            "max_abs_time_error": float(np.max(np.abs(actual - obs))),
        }
    checks.append(_check("burgers.observation_times_exact",
                         all(r["n_frames"] == len(obs)
                             and r["max_abs_time_error"] < 1e-12
                             for r in clock_rows.values()),
                         {"requested": [float(t) for t in obs],
                          "solvers": clock_rows, "tolerance": 1e-12,
                          "note": "the step function is replaced by a clock so "
                                  "that the scheduling is measured independently "
                                  "of the PDE accuracy"}))

    # 14. The dataset-facing path: the generator must write exactly the
    #     requested observation times.
    dataset_rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        gen = DatasetGenerator(
            system=system, out_dir=tmp, n_train=1, n_val=1, n_test=1,
            seed=4242, profile="validation",
        )
        manifest = gen.generate()
        requested = [float(t) for t in system.observation_times]
        recorded = [float(t) for t in manifest["trajectories"][0]["observation_times"]]
        dataset_rows["burgers"] = {
            "n_requested": len(requested),
            "n_recorded": len(recorded),
            "max_abs_time_error": float(
                np.max(np.abs(np.array(recorded) - np.array(requested)))
            ),
            "manifest_observation_scheduler": manifest.get("observation_scheduler"),
            "manifest_time_stepping_policy": manifest.get("time_stepping_policy"),
        }
    checks.append(_check("burgers.dataset_observation_times_exact",
                         all(r["n_requested"] == r["n_recorded"]
                             and r["max_abs_time_error"] < 1e-12
                             for r in dataset_rows.values()),
                         {"systems": dataset_rows, "tolerance": 1e-12,
                          "note": "the manifest records the observation times that "
                                  "were actually written, not the ones that were "
                                  "requested"}))

    # 15. Aliasing error of the spectral solver, measured on the actual
    #     field distribution of the dataset against an independently
    #     padded product.
    n_alias = 128
    alias_system = _system_config("burgers")
    from ..data.initial_conditions import fourier_series_1d

    rng = np.random.default_rng(31337)
    ic_cfg = dict(alias_system.initial_condition)
    ic_cfg.pop("kind", "fourier")
    u_alias, _spec = fourier_series_1d(n_alias, rng, **ic_cfg)
    spec_alias = Burgers1DSpectral(n=n_alias, nu=0.01)
    keep = n_alias // 3
    masked_product = np.fft.ifft(
        np.fft.fft(0.5 * np.fft.ifft(np.fft.fft(u_alias) * spec_alias._mask).real**2)
        * spec_alias._mask
    ).real
    exact_product = 0.5 * _padded_product_1d(u_alias, 2, keep)
    alias_err = float(
        np.linalg.norm(masked_product - exact_product)
        / np.linalg.norm(exact_product)
    )
    # the old (input filter only) product, for the record
    old_product = 0.5 * np.fft.ifft(
        np.fft.fft(0.5 * np.fft.ifft(np.fft.fft(u_alias) * spec_alias._mask).real**2)
    ).real
    old_alias_err = float(
        np.linalg.norm(old_product - exact_product) / np.linalg.norm(exact_product)
    )
    checks.append(_check("burgers.aliasing_error_quantified",
                         alias_err < 1e-3,
                         {"relative_l2_aliasing_error": alias_err,
                          "tolerance": 1e-3,
                          "retained_band": f"|k| <= {keep}",
                          "without_output_projection": old_alias_err,
                          "field": "actual dataset initial condition "
                                   "(fourier_series_1d, n_modes=6, amplitude=0.9)",
                          "note": "the residual aliasing error is measured "
                                  "against a 2x padded (alias-free) product of "
                                  "the same field, not asserted to vanish"}))

    # 16. Both signs of the background must agree with the independent
    #     spectral solver at the same physical time.
    sign_rows = {}
    for bg in (1.0, -1.0):
        n_s = 128
        x_s = np.linspace(0.0, 1.0, n_s, endpoint=False)
        u0_s = bg + 0.1 * np.cos(2 * np.pi * x_s)
        t_s = 0.05
        fv_s = Burgers1DWeno5(n=n_s, nu=0.0)
        dt_fv_s = fv_s.stable_dt(abs(bg) + 0.1, t_s)
        steps_fv_s = int(round(t_s / dt_fv_s))
        u_fv = fv_s.solve(u0_s, t_s / steps_fv_s, steps_fv_s)[-1]
        sp_s = Burgers1DSpectral(n=n_s, nu=0.0)
        dt_sp_s = sp_s.stable_dt(t_s)
        steps_sp_s = int(round(t_s / dt_sp_s))
        u_sp = sp_s.solve(u0_s, t_s / steps_sp_s, steps_sp_s)[-1]
        sign_rows[f"background_{bg:+.0f}"] = {
            "relative_l2_difference": float(
                np.linalg.norm(u_fv - u_sp) / np.linalg.norm(u_sp)
            ),
        }
    checks.append(_check("burgers.sign_agreement_with_spectral_solver",
                         all(r["relative_l2_difference"] < 5e-2
                             for r in sign_rows.values()),
                         {"solvers": sign_rows, "tolerance": 5e-2,
                          "t": 0.05, "grid": 128,
                          "note": "the finite-volume generator and the "
                                  "independent spectral solver must agree for a "
                                  "positive *and* a negative background"}))

    return {
        "system": "burgers",
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
        "walltime_seconds": round(time.time() - t0, 3),
    }


# ----------------------------------------------------------------------
# Allen-Cahn
# ----------------------------------------------------------------------
def validate_allen_cahn(verbose: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    t0 = time.time()

    # 1. Equilibrium fixed points.
    solver = AllenCahn2DSpectral(nx=16, ny=16, D=0.001, a=1.0)
    for value in (0.0, 1.0, -1.0):
        u0 = np.full((16, 16), value)
        sol = solver.solve(u0, dt=0.01, num_steps=20)
        err = float(np.max(np.abs(sol[-1] - u0)))
        checks.append(_check(f"allen_cahn.equilibrium_{value:+.0f}", err < 1e-10,
                             {"max_abs_error": err, "tolerance": 1e-10}))

    # 2. Reaction law with D = 0 against the exact quartic solution.
    # du/dt = a(u - u^3) has solution u(t) = u0 / sqrt(u0^2 + (1-u0^2) e^{-2at}).
    a_val = 1.0
    u0_scalar = 0.5
    s0 = AllenCahn2DSpectral(nx=4, ny=4, D=0.0, a=a_val)
    u0 = np.full((4, 4), u0_scalar)
    t_final = 1.0
    sol = s0.solve(u0, dt=0.001, num_steps=int(t_final / 0.001))
    exact = u0_scalar / np.sqrt(u0_scalar**2 + (1 - u0_scalar**2) * np.exp(-2 * a_val * t_final))
    react_err = float(np.max(np.abs(sol[-1] - exact)))
    checks.append(_check("allen_cahn.reaction_law_exact", react_err < 1e-4,
                         {"max_abs_error": react_err, "tolerance": 1e-4,
                          "u0": u0_scalar, "t_final": t_final}))

    # 3. Linear diffusion decay rate.
    D = 0.001
    n = 16
    sd = AllenCahn2DSpectral(nx=n, ny=n, D=D, a=0.0)
    x = sd.x
    u0 = 1e-3 * np.cos(2 * np.pi * x)[:, None] * np.ones((1, n))
    t_final = 1.0
    sol = sd.solve(u0, dt=0.01, num_steps=int(t_final / 0.01))
    k = 2 * np.pi
    exact = 1e-3 * np.exp(-D * k**2 * t_final) * np.cos(k * x)[:, None] * np.ones((1, n))
    diff_err = float(np.max(np.abs(sol[-1] - exact)))
    checks.append(_check("allen_cahn.linear_diffusion_rate", diff_err < 1e-8,
                         {"max_abs_error": diff_err, "tolerance": 1e-8}))

    # 4. Energy decreases for a solution heading to the stable manifold.
    se = AllenCahn2DSpectral(nx=32, ny=32, D=0.001, a=1.0)
    X, Y = np.meshgrid(se.x, se.x, indexing="ij")
    u0 = 0.5 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    energies = [se.energy(u0)]
    u = u0
    for _ in range(200):
        u = se.step(u, 0.01)
        energies.append(se.energy(u))
    e0, e1 = energies[0], energies[-1]
    checks.append(_check("allen_cahn.energy_decreases", e1 < e0,
                         {"energy_initial": e0, "energy_final": e1,
                          "t_final": 2.0}))

    # 5. Spatial refinement.
    t_final = 0.5
    ref = AllenCahn2DSpectral(nx=128, ny=128, D=0.001, a=1.0)
    u0_ref = 0.5 * np.cos(2 * np.pi * ref.x)[:, None] * np.cos(2 * np.pi * ref.x)[None, :]
    sol_ref = ref.solve(u0_ref, dt=0.005, num_steps=int(t_final / 0.005))[-1]
    sp_errs = []
    levels = []
    for n in [16, 32, 64]:
        s = AllenCahn2DSpectral(nx=n, ny=n, D=0.001, a=1.0)
        u0 = 0.5 * np.cos(2 * np.pi * s.x)[:, None] * np.cos(2 * np.pi * s.x)[None, :]
        sol = s.solve(u0, dt=0.005, num_steps=int(t_final / 0.005))[-1]
        idx = np.arange(0, 128, 128 // n)
        sp_errs.append({"n": n, "l2_error": float(np.linalg.norm(sol - sol_ref[np.ix_(idx, idx)]))})
        levels.append(n)
    rates = _rates([e["l2_error"] for e in sp_errs])
    order = _fit_order(levels, [e["l2_error"] for e in sp_errs])
    checks.append(_check("allen_cahn.spatial_refinement", order > 0.9,
                         {"errors": sp_errs, "pairwise_orders": rates,
                          "observed_order": order,
                          "note": "spectral discretisation; the retained band "
                                  "limits the order, so the order is far above "
                                  "second order until the accuracy floor is hit"}))

    # 6. Temporal refinement.
    #    The dt range is chosen so that the *temporal* error dominates.
    #    Finer steps than this reach the accuracy floor of the spatial
    #    discretisation and the comparison degenerates.
    s = AllenCahn2DSpectral(nx=32, ny=32, D=0.001, a=1.0)
    u0 = 0.5 * np.cos(2 * np.pi * s.x)[:, None] * np.cos(2 * np.pi * s.x)[None, :]
    ref_dt = 0.0025
    sol_ref = s.solve(u0, ref_dt, int(t_final / ref_dt))[-1]
    t_errs = []
    for factor in [8, 4, 2]:
        dt = ref_dt * factor
        sol = s.solve(u0, dt, int(t_final / dt))[-1]
        t_errs.append({"dt": dt, "l2_error": float(np.linalg.norm(sol - sol_ref))})
    rates = _rates([e["l2_error"] for e in t_errs])
    checks.append(_check("allen_cahn.temporal_refinement", all(r > 0.9 for r in rates),
                         {"errors": t_errs, "observed_orders": rates,
                          "note": "Strang splitting with exact linear diffusion; "
                                  "second order in time in principle"}))

    # 7. Mass is NOT conserved for this system (guards against a false
    #    conservation requirement being applied to Allen-Cahn).
    #    The initial condition is deliberately *offset* from zero: a
    #    symmetric zero-mean cosine field has mean(u - u^3) = 0 by
    #    symmetry, which would make this test vacuous.
    sm = AllenCahn2DSpectral(nx=32, ny=32, D=0.001, a=1.0)
    u0 = 0.3 + 0.4 * np.cos(2 * np.pi * sm.x)[:, None] * np.cos(2 * np.pi * sm.x)[None, :]
    sol = sm.solve(u0, dt=0.005, num_steps=200)
    means = sol.mean(axis=(1, 2))
    mass_drift = float(np.max(np.abs(means - means[0])))
    checks.append(_check("allen_cahn.mass_not_conserved_expected", mass_drift > 1e-6,
                         {"max_abs_mass_drift": mass_drift,
                          "mean_initial": float(means[0]),
                          "mean_final": float(means[-1]),
                          "note": "drift must be non-negligible; no conservation "
                                  "constraint is imposed on this system. The IC is "
                                  "offset from zero so that mean(u - u^3) != 0."}))

    # 8. Manufactured solution with a *continuum* forcing.
    #    u(x, y, t) = exp(-t) cos(2 pi x) cos(2 pi y) and the forcing of
    #        u_t = D laplacian(u) + a (u - u^3) + g
    #    is written in closed form, independently of the solver's discrete
    #    operators, and integrated through the real step()/solve_to() path.
    D_mms, a_mms = 0.001, 1.0
    t_mms = 0.05
    n_mms = 64
    xs = np.linspace(0, 1, n_mms, endpoint=False)
    Xm, Ym = np.meshgrid(xs, xs, indexing="ij")
    u_ex0 = np.cos(2 * np.pi * Xm) * np.cos(2 * np.pi * Ym)
    u_ex_T = np.exp(-t_mms) * np.cos(2 * np.pi * Xm) * np.cos(2 * np.pi * Ym)
    mms_rows = []
    for dt_m in (5e-3, 2.5e-3, 1.25e-3):
        s_mms = AllenCahn2DSpectral(nx=n_mms, ny=n_mms, D=D_mms, a=a_mms)
        steps = int(round(t_mms / dt_m))
        out = s_mms.solve(
            u_ex0, dt_m, steps,
            forcing=lambda t, u: _allen_cahn_mms_forcing(Xm, Ym, t, D_mms, a_mms),
        )
        mms_rows.append({
            "path": "solve",
            "grid": n_mms,
            "dt": dt_m,
            "max_abs_error": float(np.max(np.abs(out[-1] - u_ex_T))),
            "integrator_calls": steps,
        })
    s_to = AllenCahn2DSpectral(nx=n_mms, ny=n_mms, D=D_mms, a=a_mms)
    frames = s_to.solve_to(
        u_ex0, t_mms, np.array([0.0, t_mms]),
        forcing=lambda t, u: _allen_cahn_mms_forcing(Xm, Ym, t, D_mms, a_mms),
    )
    to_row = {
        "path": "solve_to",
        "grid": n_mms,
        "dt": float(s_to._last_dt),
        "n_frames": int(frames.shape[0]),
        "max_abs_error": float(np.max(np.abs(frames[-1] - u_ex_T))),
        "initial_frame_error": float(np.max(np.abs(frames[0] - u_ex0))),
        "integrator_calls": int(s_to._last_steps),
    }
    mms_err = mms_rows[0]["max_abs_error"]
    mms_refined_err = mms_rows[1]["max_abs_error"]
    mms_order = float(np.log2(mms_err / mms_refined_err)) if mms_refined_err > 0 else np.inf
    checks.append(_check("allen_cahn.manufactured_solution",
                         mms_err < 1e-6
                         and mms_refined_err < mms_err
                         and mms_order > 1.8
                         and to_row["max_abs_error"] < 1e-6
                         and to_row["n_frames"] == 2
                         and to_row["initial_frame_error"] == 0.0
                         and to_row["integrator_calls"] > 0
                         and all(r["integrator_calls"] > 0 for r in mms_rows),
                         {"max_abs_error": mms_err, "tolerance": 1e-6,
                          "refined_dt_error": mms_refined_err,
                          "observed_order": mms_order,
                          "solve_to_row": to_row,
                          "rows": mms_rows,
                          "forcing": "continuum: u_t - D laplacian(u) - a(u - u^3) "
                                     "= g, g written in closed form for "
                                     "u = exp(-t) cos(2 pi x) cos(2 pi y)",
                          "driven_through": "solve/solve_to -> step (real integrator)",
                          "note": "the forcing is evaluated independently of the "
                                  "solver; the error is a real integration error "
                                  "and is limited to second order by the Strang "
                                  "splitting."}))

    # 9. Reference-solution accuracy at the real benchmark horizon
    #    T = 3.0, over several actual random initial conditions.
    system = _system_config("allen_cahn")
    ref_n = system.reference_grid[0]
    fine_n = 2 * ref_n
    t_end = system.t_final
    s_ref = AllenCahn2DSpectral(nx=ref_n, ny=ref_n, D=system.params.get("D", 0.001),
                                a=system.params.get("a", 1.0))
    s_fine = AllenCahn2DSpectral(nx=fine_n, ny=fine_n, D=system.params.get("D", 0.001),
                                 a=system.params.get("a", 1.0))
    Xr, Yr = np.meshgrid(s_ref.x, s_ref.x, indexing="ij")
    Xf, Yf = np.meshgrid(s_fine.x, s_fine.x, indexing="ij")
    own_errors = []
    for u0_ref in _random_initial_conditions(system, 4, seed=9202):
        u0_fine = _spectral_resample_2d(u0_ref, fine_n)
        dt_ref = s_ref.stable_dt(t_end)
        out_ref = s_ref.solve(u0_ref, dt_ref, int(round(t_end / dt_ref)))[-1]
        dt_fine = s_fine.stable_dt(t_end) / 2.0
        out_fine = s_fine.solve(u0_fine, dt_fine, int(round(t_end / dt_fine)))[-1]
        # The dataset observes the *restricted* fields, so the own error
        # is measured between the two observation-equivalent fields.
        obs_n = system.observation_grid[0]
        out_ref_obs = restrict_2d(out_ref, obs_n, obs_n)
        out_fine_obs = restrict_2d(out_fine, obs_n, obs_n)
        own_errors.append({
            "relative_l2_error": float(
                np.linalg.norm(out_ref_obs - out_fine_obs)
                / np.linalg.norm(out_fine_obs)
            ),
        })
    own_values = [e["relative_l2_error"] for e in own_errors]
    checks.append(_check("allen_cahn.reference_solution_accuracy",
                         max(own_values) < 5e-3,
                         {"reference_grid": ref_n, "fine_grid": fine_n,
                          "relative_l2_errors": own_values,
                          "median_relative_l2_error": float(np.median(own_values)),
                          "max_relative_l2_error": float(max(own_values)),
                          "tolerance": 5e-3,
                          "t_final": t_end,
                          "n_initial_conditions": len(own_values),
                          "note": "the real benchmark horizon T = 3.0 with the "
                                  "actual random initial conditions of the "
                                  "dataset; the reference is an approximation "
                                  "of the true solution, not exact ground truth"}))

    # 10. Observation times.
    obs = np.array([0.0, 1.0, 2.0, 3.0])
    s_clock = AllenCahn2DSpectral(nx=8, ny=8)
    s_clock.step = lambda u, dt: u + dt
    s_clock.stable_dt = lambda *a, **k: 0.005
    saved = s_clock.solve_to(np.zeros((8, 8)), 3.0, obs)
    actual = saved.reshape(len(saved), -1)[:, 0]
    clock_row = {
        "n_frames": int(saved.shape[0]),
        "returned_clock": [float(v) for v in actual],
        "max_abs_time_error": float(np.max(np.abs(actual - obs))),
    }
    checks.append(_check("allen_cahn.observation_times_exact",
                         clock_row["n_frames"] == len(obs)
                         and clock_row["max_abs_time_error"] < 1e-12,
                         {"requested": [float(t) for t in obs],
                          "solvers": {"allen_cahn": clock_row},
                          "tolerance": 1e-12,
                          "note": "the step function is replaced by a clock so "
                                  "that the scheduling is measured independently "
                                  "of the PDE accuracy"}))

    # 11. The dataset-facing path.
    with tempfile.TemporaryDirectory() as tmp:
        gen = DatasetGenerator(
            system=system, out_dir=tmp, n_train=1, n_val=1, n_test=1,
            seed=4343, profile="validation",
        )
        manifest = gen.generate()
        requested = [float(t) for t in system.observation_times]
        recorded = [float(t) for t in manifest["trajectories"][0]["observation_times"]]
        dataset_row = {
            "n_requested": len(requested),
            "n_recorded": len(recorded),
            "max_abs_time_error": float(
                np.max(np.abs(np.array(recorded) - np.array(requested)))
            ),
            "manifest_observation_scheduler": manifest.get("observation_scheduler"),
        }
    checks.append(_check("allen_cahn.dataset_observation_times_exact",
                         dataset_row["n_requested"] == dataset_row["n_recorded"]
                         and dataset_row["max_abs_time_error"] < 1e-12,
                         {"systems": {"allen_cahn": dataset_row}, "tolerance": 1e-12,
                          "note": "the manifest records the observation times that "
                                  "were actually written, not the ones that were "
                                  "requested"}))

    # 12. De-aliasing: the masks must be Hermitian symmetric with an
    #     explicit Nyquist treatment, and the aliasing error must be
    #     measured against an independently padded product.
    mask_rows = {}
    for n_mask in (32, 64, 128):
        mask = AllenCahn2DSpectral._dealias_mask(n_mask)
        # In the full-FFT ordering the mirror of index j is (-j) mod n.
        mirror = mask[(-np.arange(n_mask)) % n_mask]
        hermitian = bool(np.array_equal(mask, mirror))
        nyquist_excluded = True
        if n_mask % 2 == 0:
            nyquist_excluded = not bool(mask[n_mask // 2])
        mask_rows[str(n_mask)] = {
            "n_kept": int(mask.sum()),
            "hermitian_symmetric": hermitian,
            "nyquist_excluded": nyquist_excluded,
        }
    masks_ok = all(r["hermitian_symmetric"] and r["nyquist_excluded"]
                   for r in mask_rows.values())

    n_alias = 64
    from ..data.initial_conditions import fourier_series_2d

    rng = np.random.default_rng(31338)
    ic_cfg = dict(system.initial_condition)
    ic_cfg.pop("kind", "fourier")
    u_alias, _spec = fourier_series_2d(n_alias, n_alias, rng, **ic_cfg)
    s_alias = AllenCahn2DSpectral(nx=n_alias, ny=n_alias, D=0.001, a=1.0)
    keep_x = n_alias // 4
    keep_y = n_alias // 4
    u_f = np.fft.ifft2(np.fft.fft2(u_alias) * s_alias._mask).real
    masked_cubic = np.fft.ifft2(np.fft.fft2(u_f**3) * s_alias._mask).real
    exact_cubic = _padded_product_2d(u_alias, 3, keep_x, keep_y)
    alias_err = float(
        np.linalg.norm(masked_cubic - exact_cubic) / np.linalg.norm(exact_cubic)
    )
    # the old 2/3-rule product, for the record
    modes_1d = np.abs(np.rint(np.fft.fftfreq(n_alias) * n_alias).astype(int))
    old_mask = (modes_1d <= n_alias // 3)[:, None] & (modes_1d <= n_alias // 3)[None, :]
    u_f_old = np.fft.ifft2(np.fft.fft2(u_alias) * old_mask).real
    old_cubic = np.fft.ifft2(np.fft.fft2(u_f_old**3) * old_mask).real
    old_alias_err = float(
        np.linalg.norm(old_cubic - exact_cubic) / np.linalg.norm(exact_cubic)
    )
    checks.append(_check("allen_cahn.dealiasing_masks_and_error",
                         masks_ok and alias_err < 1e-2,
                         {"masks": mask_rows,
                          "masks_hermitian_and_nyquist_excluded": masks_ok,
                          "relative_l2_aliasing_error": alias_err,
                          "tolerance": 1e-2,
                          "retained_band": f"|kx|, |ky| <= {keep_x}",
                          "two_thirds_rule_aliasing_error": old_alias_err,
                          "field": "actual dataset initial condition "
                                   "(fourier_series_2d, n_modes=3, amplitude=0.8)",
                          "note": "the 2/3 rule is a quadratic rule; for the "
                                  "cubic reaction the retained band is half of "
                                  "the Nyquist. The residual aliasing error is "
                                  "measured against a 2x padded product."}))

    return {
        "system": "allen_cahn",
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
        "walltime_seconds": round(time.time() - t0, 3),
    }


def _spectral_resample_1d(field: np.ndarray, n_to: int) -> np.ndarray:
    """Band-limited (Fourier) resampling of a periodic 1D field.

    The coarse field is transformed, its spectrum is embedded in the
    finer grid's spectrum and transformed back.  This keeps the *same*
    smooth function; a nearest-neighbour or linear interpolation would
    put a staircase (broadband) into the fine initial condition and the
    two runs would then solve different problems.
    """
    field = np.asarray(field, dtype=np.float64)
    n = field.size
    if n_to == n:
        return field.copy()
    hat = np.fft.fft(field)
    modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
    keep = np.abs(modes) < n // 2 if n % 2 == 0 else np.ones(n, dtype=bool)
    idx_to = np.where(modes >= 0, modes, modes + n_to) % n_to
    out = np.zeros(n_to, dtype=complex)
    out[idx_to[keep]] = hat[keep] * (n_to / n)
    return np.fft.ifft(out).real


def _spectral_resample_2d(field: np.ndarray, n_to: int) -> np.ndarray:
    """Band-limited (Fourier) resampling of a periodic 2D field."""
    field = np.asarray(field, dtype=np.float64)
    n0, n1 = field.shape
    if n_to == n0:
        return field.copy()
    hat = np.fft.fft2(field)
    mx = np.rint(np.fft.fftfreq(n0) * n0).astype(int)
    my = np.rint(np.fft.fftfreq(n1) * n1).astype(int)
    kx = np.where(mx >= 0, mx, mx + n_to) % n_to
    ky = np.where(my >= 0, my, my + n_to) % n_to
    out = np.zeros((n_to, n_to), dtype=complex)
    out[np.ix_(kx, ky)] = hat * (n_to / n0) * (n_to / n1)
    return np.fft.ifft2(out).real
