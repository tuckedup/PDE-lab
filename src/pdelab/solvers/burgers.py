"""1D viscous Burgers solver.

    u_t + (u^2 / 2)_x = nu * u_xx

Domain: periodic, x in [0, L). The endpoint is **not** duplicated.

Two independent discretizations are provided:

``Burgers1DFiniteVolume``
    Conservative finite volume with a MUSCL (minmod-limited) linear
    reconstruction and a Rusanov numerical flux for the advective
    flux, plus a second-order central difference for diffusion and
    SSP-RK3 time integration. This scheme is the *primary* data
    generator: it is conservative (exact mean conservation up to
    round-off), monotone, and stays well behaved once the solution
    develops shocks, which happens early at nu = 0.01.

``Burgers1DSpectral``
    Dealiased pseudo-spectral scheme with Strang splitting (exact
    linear diffusion, explicit nonlinear advection) and an exact
    integrating-factor time integrator. This is *not* used to generate
    the dataset; it exists as an **independent cross-check** of the
    finite-volume solution while the solution is still smooth.

Both expose the same interface: ``solve(u0, dt, num_steps)``.
"""

from __future__ import annotations

import numpy as np


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minmod limiter: returns the slope of smallest magnitude with the
    sign of the larger-magnitude argument."""
    out = np.zeros_like(a)
    same_sign = a * b > 0
    out[same_sign] = np.sign(a[same_sign]) * np.minimum(
        np.abs(a[same_sign]), np.abs(b[same_sign])
    )
    return out


def ssprk3_step(u: np.ndarray, dt: float, rhs) -> np.ndarray:
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
    """
    k1 = rhs(u)
    u1 = u + dt * k1
    k2 = rhs(u1)
    u2 = 0.75 * u + 0.25 * (u1 + dt * k2)
    k3 = rhs(u2)
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

    # ------------------------------------------------------------------
    # stability policy
    # ------------------------------------------------------------------
    def max_dt(self, umax: float) -> float:
        """Largest stable explicit step for the given velocity amplitude.

        Advective (Rusanov) condition: dt <= cfl * dx / (2 * max|f'|) with
        f(u) = u^2/2, f'(u) = u. The factor 2 accounts for the Rusanov
        dissipation term.
        Diffusive condition: dt <= 0.5 * dx^2 / nu.
        """
        dt_adv = self.cfl * self.dx / max(2.0 * abs(umax), 1e-12)
        dt_diff = 0.5 * self.dx**2 / self.nu if self.nu > 0 else np.inf
        return float(min(dt_adv, dt_diff))

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
        (local Lax-Friedrichs) flux. Fully periodic via np.roll.
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

    def _rhs(self, u: np.ndarray) -> np.ndarray:
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
        return -dflux_dx + self.nu * uxx

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """One SSP-RK3 step using the validated integrator."""
        return ssprk3_step(u, dt, self._rhs)

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
    ) -> np.ndarray:
        """Integrate ``num_steps`` steps of size ``dt``; return snapshots.

        Returns an array of shape ``(num_steps // save_every + 1, n)``
        including the initial condition.
        """
        u = np.asarray(u0, dtype=np.float64).copy()
        out = [u.copy()]
        for i in range(1, num_steps + 1):
            u = self.step(u, dt)
            if i % save_every == 0:
                out.append(u.copy())
        return np.stack(out)

    def solve_to(
        self, u0: np.ndarray, t_final: float, save_times: np.ndarray
    ) -> np.ndarray:
        """Integrate to ``t_final`` and return the states at ``save_times``.

        The internal integration step is chosen independently of the
        observation spacing using the CFL/stability policy.
        """
        save_times = np.asarray(save_times, dtype=np.float64)
        umax = float(np.max(np.abs(u0)))
        dt = self.stable_dt(umax, t_final)
        n_steps = int(round(t_final / dt))
        dt = t_final / n_steps

        u = np.asarray(u0, dtype=np.float64).copy()
        out = [u.copy()]
        t = 0.0
        self._last_steps = n_steps
        self._last_dt = dt
        for i in range(1, n_steps + 1):
            u = self.step(u, dt)
            t = i * dt
            while len(out) <= len(save_times) and save_times[len(out) - 1] <= t + 1e-12:
                out.append(u.copy())
        return np.stack(out[: len(save_times)])


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

    @staticmethod
    def _dealias_mask(n: int) -> np.ndarray:
        """2/3 rule mask."""
        mask = np.zeros(n, dtype=bool)
        keep = n // 3
        mask[:keep] = True
        mask[-keep:] = True
        return mask

    def _nonlinear(self, u: np.ndarray) -> np.ndarray:
        """-(u^2/2)_x with 2/3 de-aliasing."""
        u_hat = np.fft.fft(u)
        u_filt = np.fft.ifft(u_hat * self._mask).real
        flux = 0.5 * u_filt**2
        flux_hat = np.fft.fft(flux)
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

    def solve_to(
        self, u0: np.ndarray, t_final: float, save_times: np.ndarray
    ) -> np.ndarray:
        save_times = np.asarray(save_times, dtype=np.float64)
        u = np.asarray(u0, dtype=np.float64).copy()
        out = [u.copy()]
        dt = t_final / max(int(np.ceil(t_final / (0.1 * self.dx))), 1)
        n_steps = int(round(t_final / dt))
        dt = t_final / n_steps
        t = 0.0
        self._last_steps = n_steps
        self._last_dt = dt
        for i in range(1, n_steps + 1):
            u = self.step(u, dt)
            t = i * dt
            while len(out) <= len(save_times) and save_times[len(out) - 1] <= t + 1e-12:
                out.append(u.copy())
        return np.stack(out[: len(save_times)])
