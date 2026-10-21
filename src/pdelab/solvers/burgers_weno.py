"""WENO5 (Jiang-Shu) finite-volume scheme for 1D viscous Burgers.

    u_t + (u^2 / 2)_x = nu * u_xx

Why WENO5 rather than a TVD (minmod-MUSCL) scheme
--------------------------------------------------
A minmod-limited MUSCL scheme is nominally second order in smooth
regions, but the limiter introduces a non-smooth ``|u_xx|`` term into
the reconstruction. Its local truncation error then has a kink at
inflection points, and the divergence of that error degrades the
*global* L2 convergence to first order. This was measured directly
(order 1.00) before this scheme was written.

WENO5-JS reconstructs the split flux with smooth nonlinear weights, so
the reconstruction error is smooth and the scheme avoids the limiter
kink that costs the MUSCL scheme one order.

The MUSCL-Rusanov scheme in ``burgers.py`` is retained as an
*independent* discretisation for cross-validation.

Formulation
-----------
Conservative finite volume with a WENO5-JS reconstruction of the split
fluxes and SSPRK(3,3) time integration.

**Flux splitting (sign aware).**  The physical flux ``f(u) = u^2 / 2``
is *not* sign aware: ``f >= 0`` for every state, so splitting by the
sign of ``f`` produces an identically zero ``f-`` and makes every
state use a left-biased reconstruction, even where the characteristic
speed ``f'(u) = u`` is negative.  On a negative background that
reconstruction is a downwind one and the scheme *antidiffuses* (the
perturbation energy grows).  This solver therefore uses the **global
Lax-Friedrichs (Rusanov-style) split**

    f+ = 0.5 * (f + alpha * u),      f- = 0.5 * (f - alpha * u),
    alpha = max |u|   (global over the field)

for which ``f+' = 0.5 (u + alpha) >= 0`` and ``f-' = 0.5 (u - alpha)
<= 0``.  ``f+`` is reconstructed with the left-biased stencil and ``f-``
with the right-biased stencil, and the two are recombined as
``F_{i+1/2} = WENO_L(f+)_{i+1/2} + WENO_R(f-)_{i+1/2}``.  The split is
exact (``f+ + f- = f``), so a constant state stays stationary and the
spatial mean is conserved.

**Grid convention: cell averages.**  ``u[i]`` is the *cell average* of
the solution over the cell of width ``dx`` centred at ``x[i] =
i * dx``; the periodic domain is ``[0, L)`` and the endpoint is not
duplicated.  The numerical flux ``F_{i+1/2}`` lives on the face
``x[i] + dx/2`` and the divergence ``(F_{i+1/2} - F_{i-1/2}) / dx`` is
the update of cell ``i``.  Every comparison in the validation suite
uses this convention: the discrete right-hand side is compared with the
continuous right-hand side evaluated at the same cell centres, and the
cell-average sequence is treated as the uniformly sampled band-limited
function that the spectral restriction (Fourier truncation followed by
sub-sampling) assumes.

**Time-step bound.**  The semi-discrete operator is the sum of an
upwind-biased advection operator (wave speed bounded by ``alpha``) and
the explicit centred second difference for the viscous term.  A von
Neumann analysis of the linearised problem with ``A = dt * alpha / dx``
(advective CFL number) and ``B = 2 * dt * nu / dx^2`` (diffusion
number) gives, for the upwind/FTCS combination, the *combined* bound

    A <= 1 / (1 + B / A)   <=>   dt <= dx^2 / (alpha * dx + 2 * nu)

which is **not** the minimum of the separate bounds
``dt <= cfl * dx / alpha`` and ``dt <= dx^2 / (2 nu)``: whenever both
are active the combined bound is strictly smaller than either.  The
solver uses ``min(cfl * dx / alpha, dx^2 / (alpha * dx + 2 nu))``.

**De-aliasing.**  A finite-volume WENO scheme has no aliasing problem:
the nonlinear product is never formed in Fourier space.  The
pseudo-spectral cross-check solver in ``burgers.py`` and the physics
operators in ``physics/operators.py`` are the ones that need an explicit
de-aliasing policy, and they document theirs.

**Observed order.**  The reconstruction operators below are the
published WENO5-JS candidates (Jiang & Shu 1996), which are
third-order consistent for *cell averages* and are applied here to the
point-evaluated split flux ``f(u_i)``.  The resulting scheme is
therefore second order in the smooth regime (measured: 1.99 -> 2.00
over n = 32 ... 512), not fifth order; see the report of the numerical
specialist.  This is a property of the reconstruction/consistency pair,
not of the flux splitting, which is repaired here.
"""

from __future__ import annotations

import numpy as np

from .burgers import ssprk3_step
from .observation import (
    TIME_TOL,
    scheduler_manifest_entry,
    substep_plan,
    validate_save_times,
)


class Burgers1DWeno5:
    """WENO5-JS finite-volume solver for periodic viscous Burgers."""

    system = "burgers"

    def __init__(self, n: int, nu: float = 0.01, L: float = 1.0, cfl: float = 0.4,
                 epsilon: float = 1e-6) -> None:
        self.n = int(n)
        self.nu = float(nu)
        self.L = float(L)
        self.cfl = float(cfl)
        self.epsilon = float(epsilon)
        self.dx = self.L / self.n
        self.x = np.linspace(0.0, self.L, self.n, endpoint=False)
        self._last_steps = 0
        self._last_dt = 0.0
        self._last_frame_times = np.zeros(0, dtype=np.float64)

    # ------------------------------------------------------------------
    # stability policy
    # ------------------------------------------------------------------
    def max_dt(self, umax: float) -> float:
        """Largest stable explicit step for the given velocity amplitude.

        Advective (global Lax-Friedrichs split): the wave speeds of the
        split fluxes satisfy ``|f+'| , |f-'| <= alpha = max|u|``, so the
        advective CFL number is ``A = dt * alpha / dx <= 1``; a safety
        factor ``cfl`` is applied.

        Diffusive: the explicit centred second difference requires
        ``dt <= dx^2 / (2 nu)``.

        Combined: the upwind advection and the explicit diffusion do
        *not* admit the minimum of the two separate bounds.  The von
        Neumann analysis of the linearised operator gives

            dt <= dx^2 / (alpha * dx + 2 * nu)
            = 1 / (alpha / dx + 2 nu / dx^2)

        which is the binding constraint whenever the advective and
        diffusive limits are both active.
        """
        alpha = max(abs(float(umax)), 1e-12)
        dt_adv = self.cfl * self.dx / alpha
        if self.nu > 0:
            dt_combined = self.dx**2 / (alpha * self.dx + 2.0 * self.nu)
        else:
            dt_combined = np.inf
        return float(min(dt_adv, dt_combined))

    def stable_dt(self, umax: float, t_final: float, dt_max: float | None = None) -> float:
        limit = self.max_dt(umax)
        if dt_max is not None:
            limit = min(limit, dt_max)
        n_steps = max(int(np.ceil(t_final / limit)), 1)
        return t_final / n_steps

    # ------------------------------------------------------------------
    # flux splitting and reconstruction
    # ------------------------------------------------------------------
    def split_flux(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Global Lax-Friedrichs split of the Burgers flux.

        Returns ``(f_plus, f_minus, alpha)`` with

            f_plus  = 0.5 * (f + alpha * u),   f'(f_plus)  = 0.5 (u + alpha) >= 0
            f_minus = 0.5 * (f - alpha * u),   f'(f_minus) = 0.5 (u - alpha) <= 0
            f_plus + f_minus = f,              alpha = max |u|

        The split is sign aware in the *characteristic* speed
        ``f'(u) = u``, which is what the reconstruction stencils need;
        splitting by the sign of ``f`` would not be, because ``f >= 0``
        for every state.
        """
        u = np.asarray(u, dtype=np.float64)
        alpha = float(np.max(np.abs(u))) if u.size else 0.0
        f = 0.5 * u**2
        f_plus = 0.5 * (f + alpha * u)
        f_minus = 0.5 * (f - alpha * u)
        return f_plus, f_minus, alpha

    def _weno5_left(self, f: np.ndarray) -> np.ndarray:
        """Left-biased WENO5 reconstruction of the positive split flux."""
        fm2 = np.roll(f, 2)
        fm1 = np.roll(f, 1)
        fp1 = np.roll(f, -1)
        fp2 = np.roll(f, -2)

        q0 = (2.0 * fm2 - 7.0 * fm1 + 11.0 * f) / 6.0
        q1 = (-fm1 + 5.0 * f + 2.0 * fp1) / 6.0
        q2 = (2.0 * f + 5.0 * fp1 - fp2) / 6.0

        b0 = (13.0 / 12.0) * (fm2 - 2.0 * fm1 + f) ** 2 + 0.25 * (fm2 - 4.0 * fm1 + 3.0 * f) ** 2
        b1 = (13.0 / 12.0) * (fm1 - 2.0 * f + fp1) ** 2 + 0.25 * (fm1 - fp1) ** 2
        b2 = (13.0 / 12.0) * (f - 2.0 * fp1 + fp2) ** 2 + 0.25 * (3.0 * f - 4.0 * fp1 + fp2) ** 2

        a0 = 0.1 / (self.epsilon + b0) ** 2
        a1 = 0.6 / (self.epsilon + b1) ** 2
        a2 = 0.3 / (self.epsilon + b2) ** 2
        return (a0 * q0 + a1 * q1 + a2 * q2) / (a0 + a1 + a2)

    def _weno5_right(self, f: np.ndarray) -> np.ndarray:
        """Right-biased WENO5 reconstruction of the negative split flux.

        The three candidates are the mirror images of the left-biased
        ones: ``psi0 = f_{i+3}/3 - 7 f_{i+2}/6 + 11 f_{i+1}/6``,
        ``psi1 = -f_{i+2}/6 + 5 f_{i+1}/6 + f_i/3`` and
        ``psi2 = f_{i+1}/3 + 5 f_i/6 - f_{i-1}/6``.  The smoothness
        indicators are the mirrored ones as well.
        """
        fm1 = np.roll(f, 1)
        fp1 = np.roll(f, -1)
        fp2 = np.roll(f, -2)
        fp3 = np.roll(f, -3)

        p0 = (2.0 * fp3 - 7.0 * fp2 + 11.0 * fp1) / 6.0
        p1 = (-fp2 + 5.0 * fp1 + 2.0 * f) / 6.0
        p2 = (2.0 * fp1 + 5.0 * f - fm1) / 6.0

        b0 = (13.0 / 12.0) * (fp3 - 2.0 * fp2 + fp1) ** 2 + 0.25 * (fp3 - 4.0 * fp2 + 3.0 * fp1) ** 2
        b1 = (13.0 / 12.0) * (fp2 - 2.0 * fp1 + f) ** 2 + 0.25 * (fp2 - f) ** 2
        b2 = (13.0 / 12.0) * (fp1 - 2.0 * f + fm1) ** 2 + 0.25 * (3.0 * fp1 - 4.0 * f + fm1) ** 2

        a0 = 0.1 / (self.epsilon + b0) ** 2
        a1 = 0.6 / (self.epsilon + b1) ** 2
        a2 = 0.3 / (self.epsilon + b2) ** 2
        return (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2)

    def _weno5_flux(self, u: np.ndarray) -> np.ndarray:
        """WENO5 numerical flux at the right face of every cell.

        ``F[i]`` is the flux at ``x[i] + dx/2``; it is the sum of the
        left-biased reconstruction of the positive split flux and the
        right-biased reconstruction of the negative split flux.
        """
        f_plus, f_minus, _alpha = self.split_flux(u)
        return self._weno5_left(f_plus) + self._weno5_right(f_minus)

    def _rhs(self, u: np.ndarray, forcing=None, t: float = 0.0) -> np.ndarray:
        """Semi-discrete spatial operator: -(f(u))_x + nu u_xx.

        ``flux[i]`` is the flux at the face x_{i+1/2}, so the
        conservative divergence at cell centre x_i is
        ``(F_{i+1/2} - F_{i-1/2}) / dx``.

        When ``forcing`` is given it is called as ``forcing(t, u)`` and
        its value is added to the right-hand side.  It exists so that
        the manufactured-solution validation can drive the *real*
        integrator with a forcing that is evaluated independently of
        this module.
        """
        flux = self._weno5_flux(u)
        dflux_dx = (flux - np.roll(flux, 1)) / self.dx
        uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / self.dx**2
        rhs = -dflux_dx + self.nu * uxx
        if forcing is not None:
            rhs = rhs + np.asarray(forcing(t, u), dtype=np.float64)
        return rhs

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def step(self, u: np.ndarray, dt: float, forcing=None, t: float = 0.0) -> np.ndarray:
        """One SSPRK(3,3) step.

        With ``forcing`` the source is evaluated at ``t`` for the first
        stage and at ``t + dt`` for the second and third stages, which
        is the standard SSP-RK treatment of a time dependent source.
        """
        if forcing is None:
            return ssprk3_step(u, dt, self._rhs)
        return ssprk3_step(
            u, dt, lambda v, tt: self._rhs(v, forcing=forcing, t=tt), t0=t
        )

    def diffusion_only_step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Advance the viscous term alone (operator-level test hook)."""
        uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / self.dx**2
        return u + dt * self.nu * uxx

    def solve(
        self, u0: np.ndarray, dt: float, num_steps: int, save_every: int = 1,
        forcing=None, t0: float = 0.0,
    ) -> np.ndarray:
        u = np.asarray(u0, dtype=np.float64).copy()
        out = [u.copy()]
        t = float(t0)
        for i in range(1, num_steps + 1):
            if forcing is None:
                u = self.step(u, dt)
            else:
                u = self.step(u, dt, forcing=forcing, t=t)
            t += dt
            if i % save_every == 0:
                out.append(u.copy())
        return np.stack(out)

    # ------------------------------------------------------------------
    # observation-time scheduling
    # ------------------------------------------------------------------
    def _stable_step(self, u: np.ndarray, gap: float) -> float:
        """Largest stable internal step for the current state and gap."""
        umax = float(np.max(np.abs(u))) if np.size(u) else 0.0
        return float(self.stable_dt(umax, gap))

    def _advance_to(
        self, u: np.ndarray, t: float, target: float, forcing, dt_max_used: float
    ) -> tuple[np.ndarray, float, int, float]:
        """Integrate from ``t`` to ``target``, landing exactly on it."""
        n_steps = 0
        while t < target - TIME_TOL:
            gap = target - t
            dt_limit = self._stable_step(u, gap)
            n_sub, dt = substep_plan(gap, dt_limit)
            for _ in range(n_sub):
                if forcing is None:
                    u = self.step(u, dt)
                else:
                    u = self.step(u, dt, forcing=forcing, t=t)
                t += dt
            n_steps += n_sub
            dt_max_used = max(dt_max_used, dt)
        return u, target, n_steps, dt_max_used

    def solve_to(
        self, u0: np.ndarray, t_final: float, save_times: np.ndarray, forcing=None
    ) -> np.ndarray:
        """Integrate to ``t_final`` and return the states at ``save_times``.

        The returned array has exactly ``len(save_times)`` frames and
        frame ``i`` is the state at ``save_times[i]``.  The internal
        step is ``min(stable_internal_dt, next_observation_gap)``,
        sub-divided into an integer number of equal steps so that the
        integration lands exactly on the requested time.  See
        ``pdelab.solvers.observation`` for the full policy.

        ``frame_times()`` returns the actual time of every returned
        frame (they are the requested times by construction) and
        ``time_stepping_policy()`` describes the policy for the
        dataset manifest.
        """
        save_times = validate_save_times(save_times, t_final)
        u = np.asarray(u0, dtype=np.float64).copy()
        frames = [u.copy()]
        times = [0.0]
        t = 0.0
        n_steps = 0
        dt_max_used = 0.0

        for target in save_times[1:]:
            u, t, n, dt_max_used = self._advance_to(
                u, t, float(target), forcing, dt_max_used
            )
            n_steps += n
            frames.append(u.copy())
            times.append(float(target))

        if float(t_final) > t + TIME_TOL:
            u, t, n, dt_max_used = self._advance_to(
                u, t, float(t_final), forcing, dt_max_used
            )
            n_steps += n

        self._last_steps = n_steps
        self._last_dt = dt_max_used
        self._last_frame_times = np.asarray(times, dtype=np.float64)
        return np.stack(frames)

    def frame_times(self) -> np.ndarray:
        """Actual times of the frames returned by the last ``solve_to``.

        By construction these are the requested observation times; the
        accessor exists so that the dataset generator can record the
        times it actually wrote, independently of the request.
        """
        return np.asarray(self._last_frame_times, dtype=np.float64).copy()

    def time_stepping_policy(self) -> dict[str, str]:
        """Description of the time-stepping policy for the manifest."""
        policy = scheduler_manifest_entry()
        policy.update(
            {
                "solver": "burgers_weno5_js_global_lax_friedrichs",
                "time_integrator": "SSPRK(3,3)",
                "internal_step_rule": (
                    "min(stable_internal_dt, next_observation_gap), sub-divided into "
                    "ceil(gap/stable_internal_dt) equal steps so the integrator lands "
                    "exactly on the observation"
                ),
                "stability_bound": (
                    "min(cfl*dx/alpha, dx^2/(alpha*dx + 2*nu)) with alpha = max|u|; "
                    "the second term is the combined advection-diffusion von Neumann "
                    "bound and is not the minimum of the separate bounds"
                ),
            }
        )
        return policy
