"""2D diffusion-reaction (Allen-Cahn) solver.

    u_t = D * (u_xx + u_yy) + a * (u - u^3)

Domain: periodic unit square [0, 1) x [0, 1). The endpoint is not
duplicated.

Discretization
--------------
* **Space:** pseudo-spectral. The Laplacian is applied exactly in
  Fourier space, so the spatial discretisation error is spectrally small
  for the smooth fields used here.
* **Nonlinearity:** the cubic reaction term is evaluated in physical
  space from the 2/3-rule filtered field, which removes the aliasing
  that the cubic product would otherwise introduce.
* **Time:** Strang splitting. The nonlinear reaction is advanced with
  RK4 over a half step, the linear diffusion is advanced **exactly**
  with the Fourier exponential over a full step, and the reaction is
  advanced again over a half step. The scheme is second order in time
  and has no explicit diffusion stability restriction.

This is explicitly the **Allen-Cahn** two-species-free single-field
system. It is *not* Gray-Scott and no two-species dataset is involved.
"""

from __future__ import annotations

import numpy as np


class AllenCahn2DSpectral:
    """2D Allen-Cahn solver on a periodic unit square."""

    system = "allen_cahn"

    def __init__(self, nx: int, ny: int, D: float = 0.001, a: float = 1.0, L: float = 1.0) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        self.D = float(D)
        self.a = float(a)
        self.L = float(L)
        self.dx = self.L / self.nx
        self.dy = self.L / self.ny
        self.x = np.linspace(0.0, self.L, self.nx, endpoint=False)
        self.y = np.linspace(0.0, self.L, self.ny, endpoint=False)

        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        self.k2 = KX**2 + KY**2
        self._mask = self._dealias_mask(self.nx)[:, None] & self._dealias_mask(self.ny)[None, :]
        self._last_steps = 0
        self._last_dt = 0.0

    @staticmethod
    def _dealias_mask(n: int) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        keep = n // 3
        mask[:keep] = True
        mask[-keep:] = True
        return mask

    # ------------------------------------------------------------------
    def _reaction(self, u: np.ndarray) -> np.ndarray:
        """a * (u - u^3), evaluated on the 2/3-filtered field."""
        u_hat = np.fft.fft2(u)
        u_f = np.fft.ifft2(u_hat * self._mask).real
        return self.a * (u_f - u_f**3)

    def _diffuse_exact(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Exact linear diffusion step in Fourier space."""
        u_hat = np.fft.fft2(u)
        return np.fft.ifft2(u_hat * np.exp(-self.D * self.k2 * dt)).real

    def _rk4_reaction(self, u: np.ndarray, dt: float) -> np.ndarray:
        """One RK4 step of the reaction sub-problem."""
        k1 = self._reaction(u)
        k2 = self._reaction(u + 0.5 * dt * k1)
        k3 = self._reaction(u + 0.5 * dt * k2)
        k4 = self._reaction(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """One Strang-split step (half reaction, full diffusion, half reaction)."""
        u = self._rk4_reaction(u, 0.5 * dt)
        u = self._diffuse_exact(u, dt)
        u = self._rk4_reaction(u, 0.5 * dt)
        return u

    # ------------------------------------------------------------------
    def max_dt(self) -> float:
        """RK4 stability limit for the reaction sub-problem.

        For f(u) = a(u - u^3) the derivative is bounded by 3a on the
        physical range |u| <= 1 (and by 2a on the stable equilibria), so
        the RK4 limit is dt <= 2.8 / (3a). The value below is a
        conservative accuracy-driven choice rather than a hard limit.
        """
        return 2.8 / (3.0 * self.a) if self.a > 0 else np.inf

    def stable_dt(self, t_final: float, dt_max: float | None = None) -> float:
        limit = self.max_dt()
        if dt_max is not None:
            limit = min(limit, dt_max)
        n_steps = max(int(np.ceil(t_final / limit)), 1)
        return t_final / n_steps

    def solve(
        self, u0: np.ndarray, dt: float, num_steps: int, save_every: int = 1
    ) -> np.ndarray:
        """Integrate and return snapshots, shape ``(k, nx, ny)``."""
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
        """Integrate to ``t_final``, returning the states at ``save_times``."""
        save_times = np.asarray(save_times, dtype=np.float64)
        dt = self.stable_dt(t_final)
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

    # ------------------------------------------------------------------
    def energy(self, u: np.ndarray) -> float:
        """Free-energy density functional for Allen-Cahn.

        The free energy whose gradient flow generates this equation is

            F[u] = integral of [ (D/2) |grad u|^2 - a ( u^2/2 - u^4/4 ) ]

        because ``-delta F / delta u = D laplacian(u) + a (u - u^3)``,
        which is exactly the equation being integrated. The ``D/2``
        coefficient on the gradient term is essential: with a ``1/2``
        coefficient the functional is not the free energy of this
        equation and it need not decrease.

        Along a solution, dF/dt = -<u_t^2> <= 0, so F must be
        non-increasing. This is asserted by the
        ``allen_cahn.energy_decreases`` validation check.
        """
        u_hat = np.fft.fft2(u)
        grad_energy = (
            0.5 * self.D * np.mean(self.k2 * np.abs(u_hat) ** 2) / (self.nx * self.ny)
        )
        potential = -self.a * np.mean(0.5 * u**2 - 0.25 * u**4)
        return float(grad_energy + potential)
