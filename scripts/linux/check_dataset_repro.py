"""Verify that the stored dataset matches a fresh solve with the same
initial-condition seed, and locate any systematic difference."""

import json

import h5py
import numpy as np

from pdelab.data.generate import restrict_2d
from pdelab.data.initial_conditions import fourier_series_2d
from pdelab.solvers.allen_cahn import AllenCahn2DSpectral

man = json.loads(open("data/allen_cahn/allen_cahn_manifest.json").read())
syscfg = man["system"]
obs_times = np.asarray(syscfg["observation_times"])
print("t_final:", syscfg["t_final"], "n_obs_times:", len(obs_times))
print("params:", syscfg["params"], "ref grid:", syscfg["reference_grid"])

test_records = [t for t in man["trajectories"] if t["split"] == "test"]
print(f"n_test records: {len(test_records)}")

with h5py.File("data/allen_cahn/allen_cahn_test.h5", "r") as fh:
    u_test = fh["u"][:]
    seeds = fh["ic_seeds"][:]

solver = AllenCahn2DSpectral(nx=64, ny=64, D=0.001, a=1.0)

for i in [0, 1, 2]:
    rec = test_records[i]
    ic_seed = int(rec["ic_seed"])
    print(f"\n--- test traj {i}: ic_seed={ic_seed} ---")
    print("  stored ic spectrum:", {k: v for k, v in rec["ic_spectrum"].items()
                                   if k not in ("coefficients",)})

    rng = np.random.default_rng(ic_seed)
    u0, spec = fourier_series_2d(64, 64, rng, n_modes=3, amplitude=0.8, mean=0.0)
    print(f"  fresh IC: mean|u|={np.abs(u0).mean():.4f}  "
          f"stored t0: mean|u|={np.abs(u_test[i][0]).mean():.4f}")

    fresh = solver.solve_to(u0, syscfg["t_final"], obs_times)
    fresh_obs = np.stack([restrict_2d(f, 32, 32) for f in fresh])

    d0 = np.max(np.abs(fresh_obs[0] - u_test[i][0]))
    dall = np.max(np.abs(fresh_obs - u_test[i]))
    print(f"  max|fresh - stored| at t=0: {d0:.3e}   over all times: {dall:.3e}")
    print(f"  fresh  final mean|u|={np.abs(fresh_obs[-1]).mean():.4f}  "
          f"stored final mean|u|={np.abs(u_test[i][-1]).mean():.4f}")
