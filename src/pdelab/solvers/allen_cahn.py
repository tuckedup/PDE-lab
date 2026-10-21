"""2D diffusion-reaction (Allen-Cahn) solver.

    u_t = D * (u_xx + u_yy) + a * (u - u^3)

Domain: periodic unit square [0, 1) x [0, 1). The endpoint is not
duplicated.

Discretization
--------------
* **Space:** pseudo-spectral. The Laplacian is applied exactly in
  Fourier space, so the spatial discretisation error is spectrally small
  for the smooth fields used here.
* **Nonlinearity:** the *cubic* reaction term is evaluated on the
  **retained band** only.  A cubic product of a field with modes
  ``|k| <= K`` reaches ``|k| <= 3K``; with the grid Nyquist at
  ``n / 2`` the part above the Nyquist folds back into the retained
  band.  The classical alias-free rule for a cubic product is
  therefore the **1/2 rule** (``K = n // 4``, i.e. half of the
  Nyquist), *not* the 2/3 rule: with ``K = n // 3`` the cubic reaches
  ``|k| <= n`` and the folded content lands squarely inside the
  retained band.  This module therefore

  1. projects the field onto the retained band ``|k| <= n // 4`` in
     each direction before the cube is formed,
  2. forms the cube in physical space, and
  3. projects the **output** back onto the same retained band.

  The masks are Hermitian symmetric (``mask[k] == mask[n - k]``) and
  the Nyquist mode is excluded explicitly, so a real field stays real.
  The residual aliasing error is *measured* against an independently
  padded product in the validation suite rather than asserted to be
  zero.  The linear part ``a * u`` of the reaction is exact and is not
  filtered.
* **Time:** Strang splitting. The nonlinear reaction is advanced with
  RK4 over a half step, the linear diffusion is advanced **exactly**
  with the Fourier exponential over a full step, and the reaction is
  advanced again over a half step. The scheme is second order in time
  and has no explicit diffusion stability restriction.

Conventions
-----------
``u[i, j]`` is the **cell average** over the cell of width ``dx`` in
``x`` and ``dy`` in ``y`` centred at ``x[i]``, ``y[j]``.  The
pseudo-spectral operations act on that sequence; for a smooth field the
cell-average sequence and the pointwise samples differ by ``O(dx^2)``,
which is the accuracy floor of the spatial discretisation.

This is explicitly the **Allen-Cahn** two-species-free single-field
system. It is *not* Gray-Scott and no two-species dataset is involved.
"""

from __future__ import annotations

import numpy as np

from .observation import (
    TIME_TOL,
    scheduler_manifest_entry,
    substep_plan,
    validate_save_times,
)


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
        self._mask_x = self._dealias_mask(self.nx)
        self._mask_y = self._dealias_mask(self.ny)
        self._mask = self._mask_x[:, None] & self._mask_y[None, :]
        self._last_steps = 0
        self._last_dt = 0.0
        self._last_frame_times = np.zeros(0, dtype=np.float64)

    @staticmethod
    def _dealias_mask(n: int) -> np.ndarray:
        """Hermitian-symmetric 1/2-rule mask with explicit Nyquist handling.

        The mask keeps the modes ``|k| <= n // 4`` of the *full* FFT
        ordering, so ``mask[k] == mask[n - k]`` for every ``k`` (a real
        field stays real) and the Nyquist mode (its own mirror) is
        excluded for even ``n``.  Half of the Nyquist is the classical
        alias-free truncation for a *cubic* product; the 2/3 rule is
        only sufficient for a quadratic one.
        """
        if n < 4:
            raise ValueError(f"the 1/2 rule needs at least 4 points, got {n}")
        modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
        return np.abs(modes) <= (n // 4)

    # ------------------------------------------------------------------
    def _reaction(self, u: np.ndarray, forcing=None, t: float = 0.0) -> np.ndarray:
        """``a * (u - u^3)`` with an alias-free cubic product.

        The cubic is formed from the retained-band part of the field and
        the result is projected back onto the retained band.  The linear
        term ``a * u`` is exact and is not filtered.

        When ``forcing`` is given it is called as ``forcing(t, u)`` and
        added to the reaction right-hand side.  It exists so that the
        manufactured-solution validation can drive the *real* integrator
        with a forcing that is evaluated independently of this module.
        """
        u_hat = np.fft.fft2(u)
        u_f = np.fft.ifft2(u_hat * self._mask).real
        cubic_hat = np.fft.fft2(u_f**3) * self._mask
        cubic = np.fft.ifft2(cubic_hat).real
        rhs = self.a * (u - cubic)
        if forcing is not None:
            rhs = rhs + np.asarray(forcing(t, u), dtype=np.float64)
        return rhs

    def _diffuse_exact(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Exact linear diffusion step in Fourier space."""
        u_hat = np.fft.fft2(u)
        return np.fft.ifft2(u_hat * np.exp(-self.D * self.k2 * dt)).real

    def _rk4_reaction(
        self, u: np.ndarray, dt: float, forcing=None, t0: float = 0.0
    ) -> np.ndarray:
        """One RK4 step of the reaction sub-problem.

        With a time dependent source the classical RK4 stage times
        ``t0``, ``t0 + dt/2``, ``t0 + dt/2``, ``t0 + dt`` are used, which
        keeps the fourth order accuracy of the method.
        """
        if forcing is None:
            k1 = self._reaction(u)
            k2 = self._reaction(u + 0.5 * dt * k1)
            k3 = self._reaction(u + 0.5 * dt * k2)
            k4 = self._reaction(u + dt * k3)
            return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        tm = t0 + 0.5 * dt
        t1 = t0 + dt
        k1 = self._reaction(u, forcing=forcing, t=t0)
        k2 = self._reaction(u + 0.5 * dt * k1, forcing=forcing, t=tm)
        k3 = self._reaction(u + 0.5 * dt * k2, forcing=forcing, t=tm)
        k4 = self._reaction(u + dt * k3, forcing=forcing, t=t1)
        return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, u: np.ndarray, dt: float, forcing=None, t: float = 0.0) -> np.ndarray:
        """One Strang-split step (half reaction, full diffusion, half reaction).

        The optional ``forcing`` is a source term ``forcing(t, u)`` added
        to the reaction sub-problem; the two reaction half steps cover
        the time intervals ``[t, t + dt/2]`` and ``[t + dt/2, t + dt]``.
        """
        u = self._rk4_reaction(u, 0.5 * dt, forcing=forcing, t0=t)
        u = self._diffuse_exact(u, dt)
        u = self._rk4_reaction(u, 0.5 * dt, forcing=forcing, t0=t + 0.5 * dt)
        return u

    # ------------------------------------------------------------------
    def max_dt(self) -> float:
        """RK4 *stability* limit for the reaction sub-problem.

        For f(u) = a(u - u^3) the derivative is bounded by 3a on the
        physical range |u| <= 1 (and by 2a on the stable equilibria), so
        the RK4 limit is dt <= 2.8 / (3a).
        """
        return 2.8 / (3.0 * self.a) if self.a > 0 else np.inf

    def accuracy_dt(self) -> float:
        """Accuracy-driven step size for the reference solution.

        The stability limit above is far too coarse to be a *reference*:
        with dt = 0.5 the reaction is barely resolved. This value was
        chosen from the measured temporal-refinement study, in which
        dt = 0.005 gave a relative L2 error of 2.4e-7 against a
        dt = 0.0025 reference over T = 0.5. It scales with the reaction
        rate so that the same relative accuracy is obtained for a
        different ``a``.
        """
        return 0.005 / max(self.a, 1e-12)

    def stable_dt(self, t_final: float, dt_max: float | None = None) -> float:
        limit = min(self.max_dt(), self.accuracy_dt())
        if dt_max is not None:
            limit = min(limit, dt_max)
        n_steps = max(int(np.ceil(t_final / limit)), 1)
        return t_final / n_steps

    def solve(
        self, u0: np.ndarray, dt: float, num_steps: int, save_every: int = 1,
        forcing=None, t0: float = 0.0,
    ) -> np.ndarray:
        """Integrate and return snapshots, shape ``(k, nx, ny)``."""
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
        return float(self.stable_dt(gap))

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
        """Integrate to ``t_final``, returning the states at ``save_times``.

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
                "solver": "allen_cahn_spectral_strang_exact_diffusion",
                "time_integrator": "Strang splitting with exact linear diffusion",
                "internal_step_rule": (
                    "min(stable_internal_dt, next_observation_gap), sub-divided into "
                    "ceil(gap/stable_internal_dt) equal steps so the integrator lands "
                    "exactly on the observation"
                ),
                "stability_bound": "min(2.8/(3a), 0.005/a) (reaction sub-problem only)",
                "dealiasing": (
                    "1/2 rule (|k| <= n//4) on the cubic input and on the cubic "
                    "output; the 2/3 rule is not sufficient for a cubic product"
                ),
            }
        )
        return policy

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
