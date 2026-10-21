"""1D viscous Burgers solver.

    u_t + (u^2 / 2)_x = nu * u_xx

Domain: periodic, x in [0, L). The endpoint is **not** duplicated.

Three independent discretizations are provided:

``Burgers1DFiniteVolume``
    Conservative finite volume with a MUSCL (minmod-limited) linear
    reconstruction and a Rusanov numerical flux for the advective
    flux, plus a second-order central difference for diffusion and
    SSP-RK3 time integration.  The Rusanov flux is a *local* Lax
    Friedrichs flux and is therefore sign aware in the characteristic
    speed ``f'(u) = u`` by construction.  This scheme is an
    *independent* discretisation used for cross-validation.

``Burgers1DSpectral``
    Dealiased pseudo-spectral scheme with Strang splitting (exact
    linear diffusion, explicit nonlinear advection) and an exact
    integrating-factor time integrator.  This is *not* used to generate
    the dataset; it exists as an **independent cross-check** of the
    finite-volume solution while the solution is still smooth.

``Burgers1DWeno5`` (in ``burgers_weno.py``)
    The primary generator: WENO5-JS with a global Lax-Friedrichs flux
    split.

Conventions
-----------
All three schemes store **cell averages**: ``u[i]`` is the cell average
over the cell of width ``dx = L / n`` centred at ``x[i] = i * dx``.
The spectral scheme stores the same sequence on the same grid; because
a cell-average sequence of a smooth field differs from the pointwise
samples by ``O(dx^2)``, the spectral truncation used for de-aliasing
and the Fourier truncation used by the dataset restriction are
consistent with the cell-average convention to that order.

De-aliasing
-----------
The spectral solver evaluates the nonlinear flux on the **retained
band** only:

* the input field is projected onto the retained band (the 2/3 rule,
  ``|k| <= n // 3``) before the product is formed;
* the nonlinear output is projected back onto the same band.

The 2/3 rule is the classical alias-free rule for a *quadratic*
product; the product of two fields with ``|k| <= n // 3`` reaches
``|k| <= 2 n // 3`` and the part of that band above the grid Nyquist
would fold back into the retained band.  Projecting the output removes
it.  The masks are Hermitian symmetric (``mask[k] == mask[n - k]``)
and the Nyquist mode is excluded explicitly, so a real field stays
real.  The residual aliasing error is *measured* against an
independently padded product in the validation suite rather than
asserted to be zero.

Both expose the same interface: ``solve(u0, dt, num_steps)`` and
``solve_to(u0, t_final, save_times)``.
"""

from __future__ import annotations

import numpy as np

from .observation import (
    TIME_TOL,
    scheduler_manifest_entry,
    substep_plan,
    validate_save_times,
)


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod limiter: returns the slope of smallest magnitude with the
    sign of the larger-magnitude argument."""
    out = np.zeros_like(a)
    same_sign = a * b > 0
    out[same_sign] = np.sign(a[same_sign]) * np.minimum(
        np.abs(a[same_sign]), np.abs(b[same_sign])
    )
    return out


def ssprk3_step(u: np.ndarray, dt: float, rhs, t0: float | None = None) -> np.ndarray:
    """One SSP-RK3 (Shu-Osher) step of ``du/dt = rhs(u)``.

    The formulation is the standard SSPRK(3,3) recursion

        u1 = u + dt*k1
        u2 = 3/4*u + 1/4*(u1 + dt*k2)
        u3 = 1/3*u + 2/3*(u2 + dt*k3)

    which is consistent and third order; the stage arguments and the
    combination weights are not interchangeable, and an incorrect
    pairing silently reduces the method to a lower order. The
    consistency of this integrator is asserted by the dedicated
    ``burgers.time_integrator_consistency`` check in the validation
    suite, which drives it with a linear ODE whose exact solution is
    known.

    When ``t0`` is given, ``rhs`` is called as ``rhs(u, t)`` and the
    source is evaluated at ``t0`` for the first stage and at
    ``t0 + dt`` for the second and third stages.  Without ``t0`` the
    autonomous form ``rhs(u)`` is used.
    """
    if t0 is None:
        k1 = rhs(u)
        u1 = u + dt * k1
        k2 = rhs(u1)
        u2 = 0.75 * u + 0.25 * (u1 + dt * k2)
        k3 = rhs(u2)
        return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * k3)
    t1 = t0 + dt
    k1 = rhs(u, t0)
    u1 = u + dt * k1
    k2 = rhs(u1, t1)
    u2 = 0.75 * u + 0.25 * (u1 + dt * k2)
    k3 = rhs(u2, t1)
    return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * k3)


class Burgers1DFiniteVolume:
    """Conservative MUSCL-Rusanov finite-volume solver for periodic Burgers."""

    system = "burgers"

    def __init__(self, n: int, nu: float = 0.01, L: float = 1.0, cfl: float = 0.4) -> None:
        self.n = int(n)
        self.nu = float(nu)
        self.L = float(L)
        self.cfl = float(cfl)
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

        Advective (Rusanov): dt <= cfl * dx / (2 * max|f'|) with
        f(u) = u^2/2, f'(u) = u. The factor 2 accounts for the Rusanov
        dissipation term.

        Diffusive: dt <= 0.5 * dx^2 / nu.

        Combined: the upwind advection and the explicit centred
        diffusion admit the von Neumann bound
        ``dt <= dx^2 / (alpha * dx + 2 * nu)``, which is *not* the
        minimum of the two separate bounds whenever both are active.
        """
        alpha = max(abs(float(umax)), 1e-12)
        dt_adv = self.cfl * self.dx / (2.0 * alpha)
        if self.nu > 0:
            dt_combined = self.dx**2 / (alpha * self.dx + 2.0 * self.nu)
        else:
            dt_combined = np.inf
        return float(min(dt_adv, dt_combined))

    def stable_dt(self, umax: float, t_final: float, dt_max: float | None = None) -> float:
        """Choose an integer step count and return the corresponding dt."""
        dt_limit = self.max_dt(umax)
        if dt_max is not None:
            dt_limit = min(dt_limit, dt_max)
        n_steps = int(np.ceil(t_final / dt_limit))
        n_steps = max(n_steps, 1)
        return t_final / n_steps

    # ------------------------------------------------------------------
    # spatial operators
    # ------------------------------------------------------------------
    def _rusanov_flux(self, u: np.ndarray) -> np.ndarray:
        """Flux at the right face of each cell, shape (n,).

        MUSCL reconstruction with a minmod limiter followed by a Rusanov
        (local Lax-Friedrichs) flux. Fully periodic via np.roll.  The
        Rusanov dissipation is proportional to the *characteristic*
        speed ``f'(u) = u``, so this flux is sign aware.
        """
        # Slopes: forward and backward differences
        d_minus = u - np.roll(u, 1)          # u_i - u_{i-1}
        d_plus = np.roll(u, -1) - u          # u_{i+1} - u_i
        slope = _minmod(d_minus, d_plus)

        # Left and right states at face i+1/2
        u_left = u + 0.5 * slope
        u_right = np.roll(u, -1) - 0.5 * np.roll(slope, -1)

        f_left = 0.5 * u_left**2
        f_right = 0.5 * u_right**2
        alpha = np.maximum(np.abs(u_left), np.abs(u_right))
        return 0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left)

    def _rhs(self, u: np.ndarray, forcing=None, t: float = 0.0) -> np.ndarray:
        """Semi-discrete spatial operator: -(f(u))_x + nu u_xx.

        ``flux[i]`` is the flux at the face x_{i+1/2}, so the
        conservative divergence at cell centre x_i is
        ``(F_{i+1/2} - F_{i-1/2}) / dx``. Using the wrong roll here
        shifts the whole solution by one cell, which stays consistent
        but silently degrades the observed convergence to first order.
        """
        flux = self._rusanov_flux(u)
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
        """One SSP-RK3 step using the validated integrator."""
        if forcing is None:
            return ssprk3_step(u, dt, self._rhs)
        return ssprk3_step(
            u, dt, lambda v, tt: self._rhs(v, forcing=forcing, t=tt), t0=t
        )

    def diffusion_only_step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Advance the viscous term alone (operator-level test hook).

        Used to validate the second-derivative discretisation against
        the analytic decay of a single Fourier mode, independently of
        the convective term.
        """
        uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / self.dx**2
        return u + dt * self.nu * uxx

    def solve(
        self,
        u0: np.ndarray,
        dt: float,
        num_steps: int,
        save_every: int = 1,
        forcing=None,
        t0: float = 0.0,
    ) -> np.ndarray:
        """Integrate ``num_steps`` steps of size ``dt``; return snapshots.

        Returns an array of shape ``(num_steps // save_every + 1, n)``
        including the initial condition.
        """
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
        umax = float(np.max(np.abs(u))) if np.size(u) else 0.0
        return float(self.stable_dt(umax, gap))

    def _advance_to(
        self, u: np.ndarray, t: float, target: float, forcing, dt_max_used: float
    ) -> tuple[np.ndarray, float, int, float]:
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

        Exactly ``len(save_times)`` frames are returned and frame ``i``
        is the state at ``save_times[i]``.  The internal step is
        ``min(stable_internal_dt, next_observation_gap)``, sub-divided
        into an integer number of equal steps so that the integration
        lands exactly on the requested time.
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
        """Actual times of the frames returned by the last ``solve_to``."""
        return np.asarray(self._last_frame_times, dtype=np.float64).copy()

    def time_stepping_policy(self) -> dict[str, str]:
        """Description of the time-stepping policy for the manifest."""
        policy = scheduler_manifest_entry()
        policy.update(
            {
                "solver": "burgers_muscl_rusanov",
                "time_integrator": "SSPRK(3,3)",
                "internal_step_rule": (
                    "min(stable_internal_dt, next_observation_gap), sub-divided into "
                    "ceil(gap/stable_internal_dt) equal steps so the integrator lands "
                    "exactly on the observation"
                ),
                "stability_bound": (
                    "min(cfl*dx/(2*alpha), dx^2/(alpha*dx + 2*nu)) with alpha = max|u|"
                ),
            }
        )
        return policy


class Burgers1DSpectral:
    """Dealiased pseudo-spectral Burgers solver (cross-check only)."""

    system = "burgers"

    def __init__(self, n: int, nu: float = 0.01, L: float = 1.0) -> None:
        self.n = int(n)
        self.nu = float(nu)
        self.L = float(L)
        self.dx = self.L / self.n
        self.x = np.linspace(0.0, self.L, self.n, endpoint=False)
        self.k = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dx)
        self._mask = self._dealias_mask(self.n)
        self._exp = np.exp(-self.nu * self.k**2)  # per full step handled in code
        self._last_steps = 0
        self._last_dt = 0.0
        self._last_frame_times = np.zeros(0, dtype=np.float64)

    @staticmethod
    def _dealias_mask(n: int) -> np.ndarray:
        """Hermitian-symmetric 2/3-rule mask with explicit Nyquist handling.

        The mask keeps the modes ``|k| <= n // 3`` of the *full* FFT
        ordering, so ``mask[k] == mask[n - k]`` for every ``k`` (a real
        field stays real) and the Nyquist mode (which is its own
        mirror) is excluded for even ``n``.  For odd ``n`` there is no
        Nyquist mode and the same rule applies.
        """
        if n < 4:
            raise ValueError(f"the 2/3 rule needs at least 4 points, got {n}")
        modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
        return np.abs(modes) <= (n // 3)

    def _nonlinear(self, u: np.ndarray) -> np.ndarray:
        """``-(u^2/2)_x`` evaluated on the retained band.

        The input is projected onto the retained band, the product is
        formed in physical space, and the *output* is projected back
        onto the same band.  Without the output projection the product
        carries content above the retained band that folds back into
        it (the retained band reaches ``2 n // 3``, which is above the
        grid Nyquist).
        """
        u_hat = np.fft.fft(u)
        u_filt = np.fft.ifft(u_hat * self._mask).real
        flux = 0.5 * u_filt**2
        flux_hat = np.fft.fft(flux) * self._mask
        return np.fft.ifft(-1j * self.k * flux_hat).real

    def _rk4_half(self, u: np.ndarray, dt: float) -> np.ndarray:
        """One RK4 step of the explicit nonlinear part."""
        k1 = self._nonlinear(u)
        k2 = self._nonlinear(u + 0.5 * dt * k1)
        k3 = self._nonlinear(u + 0.5 * dt * k2)
        k4 = self._nonlinear(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Strang splitting: half nonlinear, full exact diffusion, half nonlinear."""
        u = self._rk4_half(u, 0.5 * dt)
        u_hat = np.fft.fft(u) * np.exp(-self.nu * self.k**2 * dt)
        u = np.fft.ifft(u_hat).real
        u = self._rk4_half(u, 0.5 * dt)
        return u

    def solve(
        self, u0: np.ndarray, dt: float, num_steps: int, save_every: int = 1
    ) -> np.ndarray:
        u = np.asarray(u0, dtype=np.float64).copy()
        out = [u.copy()]
        for i in range(1, num_steps + 1):
            u = self.step(u, dt)
            if i % save_every == 0:
                out.append(u.copy())
        return np.stack(out)

    # ------------------------------------------------------------------
    # observation-time scheduling
    # ------------------------------------------------------------------
    def stable_dt(self, t_final: float, dt_max: float | None = None) -> float:
        """Internal step for the Strang split.

        The diffusion is exact, so only the explicit nonlinear half
        step restricts the step size; the historical ``0.1 * dx``
        advective limit is kept.
        """
        limit = 0.1 * self.dx
        if dt_max is not None:
            limit = min(limit, dt_max)
        n_steps = max(int(np.ceil(t_final / limit)), 1)
        return t_final / n_steps

    def _stable_step(self, u: np.ndarray, gap: float) -> float:
        return float(self.stable_dt(gap))

    def _advance_to(
        self, u: np.ndarray, t: float, target: float, dt_max_used: float
    ) -> tuple[np.ndarray, float, int, float]:
        n_steps = 0
        while t < target - TIME_TOL:
            gap = target - t
            dt_limit = self._stable_step(u, gap)
            n_sub, dt = substep_plan(gap, dt_limit)
            for _ in range(n_sub):
                u = self.step(u, dt)
                t += dt
            n_steps += n_sub
            dt_max_used = max(dt_max_used, dt)
        return u, target, n_steps, dt_max_used

    def solve_to(
        self, u0: np.ndarray, t_final: float, save_times: np.ndarray
    ) -> np.ndarray:
        """Integrate to ``t_final`` and return the states at ``save_times``."""
        save_times = validate_save_times(save_times, t_final)
        u = np.asarray(u0, dtype=np.float64).copy()
        frames = [u.copy()]
        times = [0.0]
        t = 0.0
        n_steps = 0
        dt_max_used = 0.0

        for target in save_times[1:]:
            u, t, n, dt_max_used = self._advance_to(u, t, float(target), dt_max_used)
            n_steps += n
            frames.append(u.copy())
            times.append(float(target))

        if float(t_final) > t + TIME_TOL:
            u, t, n, dt_max_used = self._advance_to(u, t, float(t_final), dt_max_used)
            n_steps += n

        self._last_steps = n_steps
        self._last_dt = dt_max_used
        self._last_frame_times = np.asarray(times, dtype=np.float64)
        return np.stack(frames)

    def frame_times(self) -> np.ndarray:
        """Actual times of the frames returned by the last ``solve_to``."""
        return np.asarray(self._last_frame_times, dtype=np.float64).copy()

    def time_stepping_policy(self) -> dict[str, str]:
        """Description of the time-stepping policy for the manifest."""
        policy = scheduler_manifest_entry()
        policy.update(
            {
                "solver": "burgers_spectral_strang_dealiased",
                "time_integrator": "Strang splitting with exact linear diffusion",
                "internal_step_rule": (
                    "min(stable_internal_dt, next_observation_gap), sub-divided into "
                    "ceil(gap/stable_internal_dt) equal steps so the integrator lands "
                    "exactly on the observation"
                ),
                "stability_bound": "0.1 * dx (explicit nonlinear half step only)",
                "dealiasing": "2/3 rule on the input and on the nonlinear output",
            }
        )
        return policy
