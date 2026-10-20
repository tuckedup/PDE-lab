"""Diagnostic for the Allen-Cahn energy and mass checks."""

import numpy as np

from pdelab.solvers.allen_cahn import AllenCahn2DSpectral

se = AllenCahn2DSpectral(nx=32, ny=32, D=0.001, a=1.0)
X, Y = np.meshgrid(se.x, se.x, indexing="ij")
u0 = 0.5 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)

print("initial: max|u| =", np.abs(u0).max(), " mean =", u0.mean())
u = u0.copy()
for i in range(200):
    u = se.step(u, 0.01)
    if i % 25 == 0 or i == 199:
        # breakdown of the energy functional
        u_hat = np.fft.fft2(u)
        grad = 0.5 * np.mean(se.k2 * np.abs(u_hat) ** 2) / (se.nx * se.ny)
        pot = -se.a * np.mean(0.5 * u**2 - 0.25 * u**4)
        print(
            f"  step {i:4d} t={0.01*(i+1):.2f} max|u|={np.abs(u).max():.4f} "
            f"mean={u.mean():+.6e} grad={grad:.4f} pot={pot:.4f} F={grad+pot:.4f}"
        )

print()
print("Parseval check: mean(u^2) =", np.mean(u0**2))
u_hat = np.fft.fft2(u0)
print("  sum|u_hat|^2/(N^2)  =", np.sum(np.abs(u_hat) ** 2) / (32 * 32))
print("  mean(k2|u_hat|^2)   =", np.mean(se.k2 * np.abs(u_hat) ** 2))
print("  /N^2                =", np.mean(se.k2 * np.abs(u_hat) ** 2) / (32 * 32))
print("  -> correct <|grad u|^2> =", np.mean(se.k2 * np.abs(u_hat) ** 2) / (32 * 32))
print("  0.5 * that          =", 0.5 * np.mean(se.k2 * np.abs(u_hat) ** 2) / (32 * 32))

print()
print("mass drift for a phase-shifted IC (breaks the symmetry):")
u0b = 0.5 * np.cos(2 * np.pi * X + 0.7) * np.cos(2 * np.pi * Y + 0.3)
s2 = AllenCahn2DSpectral(nx=32, ny=32, D=0.001, a=1.0)
sol = s2.solve(u0b, dt=0.01, num_steps=100)
means = sol.mean(axis=(1, 2))
print("  max |mass drift| =", np.max(np.abs(means - means[0])))
