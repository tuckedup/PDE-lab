"""Is the Allen-Cahn energy growth a solver bug or under-resolution?

The Allen-Cahn interface width scales as sqrt(D/a). With D=0.001, a=1 that
is 0.0316, which is only ~1 cell on a 32x32 grid. If the solver is correct,
refining the grid must restore the expected energy decrease.
"""

import numpy as np

from pdelab.solvers.allen_cahn import AllenCahn2DSpectral


def energy(se, u):
    u_hat = np.fft.fft2(u)
    grad = 0.5 * np.mean(se.k2 * np.abs(u_hat) ** 2) / (se.nx * se.ny)
    pot = -se.a * np.mean(0.5 * u**2 - 0.25 * u**4)
    return grad + pot


for D in [0.001, 0.01]:
    print(f"=== D={D}, a=1, T=1.0 ===")
    for n in [32, 64, 128, 256]:
        se = AllenCahn2DSpectral(nx=n, ny=n, D=D, a=1.0)
        X, Y = np.meshgrid(se.x, se.x, indexing="ij")
        u0 = 0.5 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        u = u0.copy()
        for _ in range(100):  # t = 1.0 at dt=0.01
            u = se.step(u, 0.01)
        e0, e1 = energy(se, u0), energy(se, u)
        width = np.sqrt(D / se.a)
        print(
            f"  n={n:4d} dx={1/n:.4f} interface_cells={width*n:.1f} "
            f"F0={e0:+.4f} F1={e1:+.4f} delta={e1-e0:+.4f} max|u|={np.abs(u).max():.4f}"
        )
    print()

print("=== mass test with an offset IC (breaks the cos symmetry) ===")
for n in [32, 64]:
    se = AllenCahn2DSpectral(nx=n, ny=n, D=0.01, a=1.0)
    X, Y = np.meshgrid(se.x, se.x, indexing="ij")
    u0 = 0.3 + 0.4 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sol = se.solve(u0, dt=0.005, num_steps=200)
    means = sol.mean(axis=(1, 2))
    print(f"  n={n:4d} mean0={means[0]:+.6f} meanT={means[-1]:+.6f} "
          f"drift={np.max(np.abs(means-means[0])):.6e}")
