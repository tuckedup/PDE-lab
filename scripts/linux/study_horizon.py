"""How do the trivial baseline and the task difficulty scale with the horizon?

The prompt requires a meaningful prediction horizon: copying the input
must NOT nearly meet the accuracy goal. This measures the persistence
nRMSE on the *actual dataset initial-condition distribution* for
candidate horizons, so the horizon is chosen on evidence rather than by
assertion.

The persistence forecast is built with ``np.repeat(initial, n_steps,
axis=1)`` on a **batched** array of shape (n_traj, n_times, ...), so
that axis 1 is the time axis. (On a single trajectory, axis 1 is a
spatial axis and the same expression tiles the wrong axis - this was a
real bug in an earlier version of this script.)
"""

import numpy as np

from pdelab.data.initial_conditions import fourier_series_2d
from pdelab.evaluation.metrics import nrmse_per_trajectory
from pdelab.solvers.allen_cahn import AllenCahn2DSpectral


def persistence_nrmse_batched(traj: np.ndarray) -> float:
    """Mean per-trajectory persistence nRMSE for a (n_traj, n_times, ...) array."""
    ref = traj[:, 1:]
    pred = np.repeat(traj[:, :1], ref.shape[1], axis=1)
    return float(np.mean(nrmse_per_trajectory(pred, ref)))


print(f"{'T':>6} {'dt_obs':>7} {'nsnap':>6} {'persist_nRMSE':>15} {'mean|du|/|u|':>15}")
for T, dt_obs in [(1.0, 0.05), (2.0, 0.05), (3.0, 0.1), (5.0, 0.1), (5.0, 0.05)]:
    obs_times = np.arange(0.0, T + 1e-9, dt_obs)
    trajs = []
    rel_change = []
    for seed in range(6):
        rng = np.random.default_rng(2000 + seed)
        s = AllenCahn2DSpectral(nx=64, ny=64, D=0.001, a=1.0)
        u0, _ = fourier_series_2d(64, 64, rng, n_modes=3, amplitude=0.8)
        trajs.append(s.solve_to(u0, T, obs_times))
        rel_change.append(
            float(np.mean(np.abs(trajs[-1][-1] - trajs[-1][0]))
                  / (np.mean(np.abs(trajs[-1][0])) + 1e-12))
        )
    traj = np.stack(trajs)
    print(
        f"{T:>6} {dt_obs:>7} {len(obs_times):>6} {persistence_nrmse_batched(traj):>15.4f} "
        f"{np.mean(rel_change):>15.4f}"
    )
