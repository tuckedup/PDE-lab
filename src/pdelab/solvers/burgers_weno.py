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
the reconstruction error is smooth and the scheme retains its formal
order in smooth regions while remaining non-oscillatory at shocks.

The MUSCL-Rusanov scheme in ``burgers.py`` is retained as an
*independent* discretisation for cross-validation.

Formulation
-----------
Conservative finite volume with a WENO5 reconstruction of the split
fluxes f+/- = max/min(f, 0) with f(u) = u^2/2, second-order central
differences for the viscous term, and SSPRK(3,3) time integration.
"""

from __future__ import annotations

import numpy as np

from .burgers import ssprk3_step


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

    # ------------------------------------------------------------------
    def max_dt(self, umax: float) -> float:
        """Largest stable explicit step.

        Advective: dt <= cfl * dx / max|f'(u)| with f'(u) = u.
        Diffusive: dt <= 0.5 * dx^2 / nu (explicit central diffusion).
        """
        dt_adv = self.cfl * self.dx / max(abs(umax), 1e-12)
        dt_diff = 0.5 * self.dx**2 / self.nu if self.nu > 0 else np.inf
        return float(min(dt_adv, dt_diff))

    def stable_dt(self, umax: float, t_final: float, dt_max: float | None = None) -> float:
        limit = self.max_dt(umax)
        if dt_max is not None:
            limit = min(limit, dt_max)
        n_steps = max(int(np.ceil(t_final / limit)), 1)
        return t_final / n_steps

    # ------------------------------------------------------------------
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
        """Right-biased WENO5 reconstruction of the negative split flux."""
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
        """WENO5 numerical flux at the right face of every cell."""
        f = 0.5 * u**2
        f_plus = np.where(f > 0.0, f, 0.0)
        f_minus = np.where(f < 0.0, f, 0.0)
        return self._weno5_left(f_plus) + self._weno5_right(f_minus)

    def _rhs(self, u: np.ndarray) -> np.ndarray:
        """Semi-discrete spatial operator: -(f(u))_x + nu u_xx.

        ``flux[i]`` is the flux at the face x_{i+1/2}, so the
        conservative divergence at cell centre x_i is
        ``(F_{i+1/2} - F_{i-1/2}) / dx``.
        """
        flux = self._weno5_flux(u)
        dflux_dx = (flux - np.roll(flux, 1)) / self.dx
        uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / self.dx**2
        return -dflux_dx + self.nu * uxx

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """One SSPRK(3,3) step."""
        return ssprk3_step(u, dt, self._rhs)

    def diffusion_only_step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Advance the viscous term alone (operator-level test hook)."""
        uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / self.dx**2
        return u + dt * self.nu * uxx

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
        umax = float(np.max(np.abs(u0)))
        dt = self.stable_dt(umax, t_final)
        n_steps = max(int(round(t_final / dt)), 1)
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
