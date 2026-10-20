"""Test the aliasing hypothesis for the Allen-Cahn cubic nonlinearity.

The 2/3 rule is alias-free for a *quadratic* product. For a cubic, the
retained band must satisfy 3*K <= N/2, i.e. K <= N/6, for the product to
be alias-free. This script runs the same dynamics with a 1/6 band and
compares the free-energy behaviour.
"""

import numpy as np

from pdelab.solvers.allen_cahn import AllenCahn2DSpectral


def energy(se, u):
    u_hat = np.fft.fft2(u)
    grad = 0.5 * np.mean(se.k2 * np.abs(u_hat) ** 2) / (se.nx * se.ny)
    pot = -se.a * np.mean(0.5 * u**2 - 0.25 * u**4)
    return grad + pot


class AC(AllenCahn2DSpectral):
    def __init__(self, *a, keep_frac=3, **kw):
        super().__init__(*a, **kw)
        n = self.nx
        keep = int(n // keep_frac)
        mask = np.zeros(n, dtype=bool)
        mask[:keep] = True
        mask[-keep:] = True
        self._mask = mask[:, None] & mask[None, :]


for D in [0.001, 0.01]:
    print(f"=== D={D} ===")
    for keep_frac in [3, 6]:
        se = AC(nx=64, ny=64, D=D, a=1.0, keep_frac=keep_frac)
        X, Y = np.meshgrid(se.x, se.x, indexing="ij")
        u0 = 0.5 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        u = u0.copy()
        for _ in range(100):
            u = se.step(u, 0.01)
        e0, e1 = energy(se, u0), energy(se, u)
        # also report the discrete energy-balance residual: dF/dt should be
        # -(1/D) * mean(u_t^2)
        u_prev = u
        u_next = se.step(u, 0.01)
        dF = energy(se, u_next) - energy(se, u_prev)
        ut = (u_next - u_prev) / 0.01
        print(
            f"  keep=N/{keep_frac}: F0={e0:+.4f} F1={e1:+.4f} dF={e1-e0:+.4f} "
            f"| -(1/D)*<ut^2> = {-(1.0/D)*np.mean(ut**2):+.4f} max|u|={np.abs(u).max():.4f}"
        )
    print()
