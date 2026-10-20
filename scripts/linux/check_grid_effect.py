"""Compare the persistence baseline on the reference grid and its
restriction to the observation grid, for the same initial condition."""

import numpy as np

from pdelab.data.generate import restrict_2d
from pdelab.data.initial_conditions import fourier_series_2d
from pdelab.solvers.allen_cahn import AllenCahn2DSpectral


def persistence_nrmse(traj: np.ndarray) -> float:
    ref = traj[:, 1:]
    pred = np.repeat(traj[:, :1], ref.shape[1], axis=1)
    num = np.sum((pred - ref) ** 2)
    den = np.sum(ref**2) + 1e-12
    return float(np.sqrt(num / den))


T = 1.0
obs_times = np.arange(0.0, T + 1e-9, 0.05)

for seed in range(4):
    rng = np.random.default_rng(2000 + seed)
    s64 = AllenCahn2DSpectral(nx=64, ny=64, D=0.001, a=1.0)
    u0, _ = fourier_series_2d(64, 64, rng, n_modes=3, amplitude=0.8)
    traj64 = s64.solve_to(u0, T, obs_times)
    traj32 = np.stack([restrict_2d(f, 32, 32) for f in traj64])

    print(
        f"seed {seed}: persist(64x64)={persistence_nrmse(traj64):.4f}  "
        f"persist(32x32)={persistence_nrmse(traj32):.4f}  "
        f"mean|u|64={np.abs(traj64).mean():.4f} mean|u|32={np.abs(traj32).mean():.4f}  "
        f"max|u|64={np.abs(traj64).max():.4f} max|u|32={np.abs(traj32).max():.4f}"
    )

    # spectrum of the final state
    u_hat = np.fft.fft2(traj64[-1])
    mag = np.abs(u_hat)
    tot = mag.sum()
    kx = np.fft.fftfreq(64) * 64
    ky = np.fft.fftfreq(64) * 64
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kmag = np.sqrt(KX**2 + KY**2)
    for band in [(0, 4), (4, 8), (8, 16), (16, 32), (32, 100)]:
        sel = (kmag >= band[0]) & (kmag < band[1])
        print(f"    k in [{band[0]},{band[1]}): {100*mag[sel].sum()/tot:5.1f}% of spectrum")
