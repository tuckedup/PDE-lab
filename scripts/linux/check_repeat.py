"""Decisive micro-test of the persistence forecast construction."""

import h5py
import numpy as np

with h5py.File("data/allen_cahn/allen_cahn_test.h5", "r") as fh:
    u = fh["u"][:]  # (16, 21, 32, 32)

print("u.shape =", u.shape)

# Case A: batched
ref_a = u[:, 1:]
pred_a = np.repeat(u[:, :1], ref_a.shape[1], axis=1)
print("A: pred_a.shape =", pred_a.shape, " ref_a.shape =", ref_a.shape)
print("   pred_a[0,0,0,:4] =", pred_a[0, 0, 0, :4])
print("   u[0,0,0,:4]      =", u[0, 0, 0, :4])
print("   pred_a[0,1,0,:4] =", pred_a[0, 1, 0, :4])
print("   pred_a[0,19,0,:4]=", pred_a[0, 19, 0, :4])
err_a = np.sum((pred_a - ref_a) ** 2, axis=(1, 2, 3))
den_a = np.sum(ref_a**2, axis=(1, 2, 3))
print("   A traj0: err=", err_a[0], " den=", den_a[0], " nRMSE=", np.sqrt(err_a[0]/den_a[0]))

# Case B: single trajectory
traj = u[0]
ref_b = traj[:, 1:]
pred_b = np.repeat(traj[:, :1], ref_b.shape[1], axis=1)
print("\nB: pred_b.shape =", pred_b.shape, " ref_b.shape =", ref_b.shape)
print("   pred_b[0,0,:4]  =", pred_b[0, 0, :4])
print("   pred_b[1,0,:4]  =", pred_b[1, 0, :4])
print("   u[0,0,0,:4]     =", u[0, 0, 0, :4])
err_b = np.sum((pred_b - ref_b) ** 2)
den_b = np.sum(ref_b**2)
print("   B traj0: err=", err_b, " den=", den_b, " nRMSE=", np.sqrt(err_b/den_b))

# Case C: tiled explicitly
pred_c = np.tile(u[:, :1], (1, ref_a.shape[1], 1, 1))
err_c = np.sum((pred_c - ref_a) ** 2, axis=(1, 2, 3))
print("\nC (tile): traj0 nRMSE =", np.sqrt(err_c[0] / den_a[0]))
print("   pred_c[0,0,0,:4] =", pred_c[0, 0, 0, :4])
print("   pred_c[0,1,0,:4] =", pred_c[0, 1, 0, :4])
