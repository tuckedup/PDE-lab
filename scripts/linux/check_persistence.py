"""Check the persistence baseline on the actual test data.

The evaluation reports a per-trajectory mean nRMSE; the horizon study
reports a global (pooled) nRMSE. Compare both on the real dataset.
"""

import numpy as np

from pdelab.data.dataset import TrajectoryDataset, compute_normalization_stats, split_paths
from pdelab.evaluation.metrics import nrmse_per_trajectory

paths = split_paths("data/allen_cahn", "allen_cahn")
stats = compute_normalization_stats(paths["train"])
ds = TrajectoryDataset(paths["test"], stats, history=1)

u = ds.u  # (n_traj, n_times, X, Y)
n_traj, n_times = u.shape[0], u.shape[1]
print(f"test trajectories: {n_traj}, snapshots: {n_times}, grid: {u.shape[2:]}")
print(f"field amplitude: mean|u| = {np.abs(u).mean():.4f}, max|u| = {np.abs(u).max():.4f}")
print(f"per-trajectory mean|u| = {np.abs(u).mean(axis=(1,2,3)).round(4)}")

ref = u[:, 1:]
pred = np.repeat(u[:, :1], ref.shape[1], axis=1)

per_traj = nrmse_per_trajectory(pred, ref)
print(f"\nper-trajectory nRMSE: {per_traj.round(4)}")
print(f"  mean = {per_traj.mean():.4f}, median = {np.median(per_traj):.4f}")

num = np.sum((pred - ref) ** 2)
den = np.sum(ref**2) + 1e-12
print(f"global (pooled) nRMSE = {np.sqrt(num/den):.4f}")

print("\nper-trajectory reference energy (sum of squares):")
print(f"  {np.sum(ref**2, axis=(1,2,3)).round(3)}")
print("per-trajectory error energy:")
print(f"  {np.sum((pred-ref)**2, axis=(1,2,3)).round(3)}")
