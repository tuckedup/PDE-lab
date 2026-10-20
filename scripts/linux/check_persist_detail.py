"""Compare persistence per-snapshot between a stored dataset trajectory
and a freshly generated one, to locate the discrepancy."""

import json

import h5py
import numpy as np

from pdelab.data.initial_conditions import fourier_series_2d
from pdelab.solvers.allen_cahn import AllenCahn2DSpectral

man = json.loads(open("data/allen_cahn/allen_cahn_manifest.json").read())
obs_times = np.asarray(man["system"]["observation_times"])

with h5py.File("data/allen_cahn/allen_cahn_test.h5", "r") as fh:
    u_test = fh["u"][:]

solver = AllenCahn2DSpectral(nx=64, ny=64, D=0.001, a=1.0)
rng = np.random.default_rng(2000)
u0, _ = fourier_series_2d(64, 64, rng, n_modes=3, amplitude=0.8, mean=0.0)
fresh = solver.solve_to(u0, 1.0, obs_times)


def report(name, traj):
    ref = traj[:, 1:]
    pred = np.repeat(traj[:, :1], ref.shape[1], axis=1)
    num = np.sum((pred - ref) ** 2)
    den = np.sum(ref**2) + 1e-12
    per_snap = np.sqrt(
        np.sum((pred - ref) ** 2, axis=(1, 2)) / (np.sum(ref**2, axis=(1, 2)) + 1e-12)
    )
    print(f"{name}: persistence nRMSE = {np.sqrt(num/den):.4f}")
    print(f"   mean|u| per snapshot: {np.abs(traj).mean(axis=(1, 2)).round(3)}")
    print(f"   persistence err per snapshot: {per_snap.round(3)}")
    print(f"   sum ref^2 = {den:.1f}   sum err^2 = {num:.1f}")


report("stored test traj 0", u_test[0])
report("fresh (seed 2000)    ", fresh)
