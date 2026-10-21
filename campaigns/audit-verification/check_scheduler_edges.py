"""Edge cases of the observation scheduler."""

import numpy as np

from pdelab.solvers.allen_cahn import AllenCahn2DSpectral
from pdelab.solvers.burgers import Burgers1DFiniteVolume, Burgers1DSpectral
from pdelab.solvers.burgers_weno import Burgers1DWeno5

print("1. a single observation time")
for solver, u0, t_final in [
    (Burgers1DWeno5(n=32), np.zeros(32), 0.1),
    (AllenCahn2DSpectral(nx=8, ny=8), np.zeros((8, 8)), 1.0),
]:
    out = solver.solve_to(u0, t_final, np.array([0.0]))
    print(f"   {type(solver).__name__}: frames={out.shape[0]} "
          f"frame_times={solver.frame_times().tolist()} "
          f"(the state is still advanced to t_final: "
          f"{solver._last_steps} internal steps)")

print("2. the last observation is earlier than t_final")
s = Burgers1DWeno5(n=64, nu=0.01)
out = s.solve_to(np.cos(2 * np.pi * s.x), 0.5, np.array([0.0, 0.1, 0.2]))
print(f"   frames={out.shape[0]} frame_times={s.frame_times().tolist()} "
      f"internal steps={s._last_steps}")

print("3. an observation spacing smaller than the internal step")
s = Burgers1DSpectral(n=64, nu=0.01)
out = s.solve_to(np.cos(2 * np.pi * s.x), 0.05,
                 np.array([0.0, 0.001, 0.002, 0.05]))
print(f"   frames={out.shape[0]} frame_times={s.frame_times().tolist()}")
assert np.allclose(s.frame_times(), [0.0, 0.001, 0.002, 0.05], atol=1e-12)

print("4. the same request twice gives the same answer (determinism)")
a = Burgers1DFiniteVolume(n=64, nu=0.01)
b = Burgers1DFiniteVolume(n=64, nu=0.01)
req = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
ra = a.solve_to(np.cos(2 * np.pi * a.x), 0.2, req)
rb = b.solve_to(np.cos(2 * np.pi * b.x), 0.2, req)
print(f"   max |difference| = {np.max(np.abs(ra - rb)):.3e}")
assert np.array_equal(ra, rb)
print("   OK")
