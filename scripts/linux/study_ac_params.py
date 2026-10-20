"""Parameter study for the 2D Allen-Cahn benchmark.

Choosing D and the horizon T is a *configuration* decision that has to
be made on evidence:

* The interface width of Allen-Cahn scales as sqrt(D/a). It must be
  resolved by the observation grid, otherwise the "reference" is
  grid-dependent and the benchmark is ill-posed.
* The dynamics must be non-trivial: the persistence baseline must be
  clearly worse than the target, otherwise the task is trivial.
* The cost of generating the reference must fit the local budget.

This script measures, for candidate (D, T, dt_obs) settings:
  - the interface width in observation cells,
  - the grid sensitivity of the solution (n=32 vs n=64),
  - the persistence nRMSE over the horizon,
  - the wall-clock cost per trajectory.
"""

import time

import numpy as np

from pdelab.solvers.allen_cahn import AllenCahn2DSpectral


def persistence_nrmse(traj: np.ndarray) -> float:
    """nRMSE of copying the initial snapshot across the whole horizon."""
    ref = traj[:, 1:]
    pred = np.repeat(traj[:, :1], ref.shape[1], axis=1)
    num = np.sum((pred - ref) ** 2)
    den = np.sum(ref**2) + 1e-12
    return float(np.sqrt(num / den))


print(f"{'D':>7} {'T':>5} {'dtobs':>6} {'iface_cells(32)':>17} {'grid_sens':>10} "
      f"{'persist_nRMSE':>15} {'max|u|T':>9} {'sec/traj':>9}")
for D, T, dt_obs in [
    (0.001, 1.0, 0.05),
    (0.001, 5.0, 0.25),
    (0.001, 10.0, 0.5),
    (0.01, 5.0, 0.25),
    (0.01, 10.0, 0.5),
]:
    obs_times = np.arange(0.0, T + 1e-9, dt_obs)
    n_snap = len(obs_times)

    def make(n, D):
        s = AllenCahn2DSpectral(nx=n, ny=n, D=D, a=1.0)
        X, Y = np.meshgrid(s.x, s.x, indexing="ij")
        u0 = 0.8 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        return s, u0

    t0 = time.time()
    s64, u0_64 = make(64, D)
    traj = s64.solve_to(u0_64, T, obs_times)
    sec = time.time() - t0

    s32, u0_32 = make(32, D)
    traj32 = s32.solve_to(u0_32, T, obs_times)
    grid_sens = float(
        np.linalg.norm(traj32[1:] - traj[1:, ::2, ::2])
        / np.linalg.norm(traj[1:, ::2, ::2])
    )
    iface_cells = np.sqrt(D / 1.0) * 32
    print(f"{D:>7} {T:>5} {dt_obs:>6} {iface_cells:>17.1f} {grid_sens:>10.4f} "
          f"{persistence_nrmse(traj):>15.4f} {np.abs(traj[-1]).max():>9.3f} {sec:>9.2f}")
