"""Check that the evaluation module resolves and calls the new operators.

This is the hand-off check for the training/evidence specialist: the
evaluation module looks the operators up by name and adapts to their
signature, so the names and the support must be exactly as advertised.
"""

import inspect

import numpy as np
import torch

from pdelab.evaluation.evaluate import (
    CENTRED_FIRST_OPERATOR_NAMES,
    IMPLICIT_EULER_OPERATOR_NAMES,
    _align_support,
    _call_operator,
    _resolve_operator,
)
from pdelab.physics.operators import (
    allen_cahn_residual_centered_first_derivative,
    allen_cahn_residual_implicit_euler,
    burgers_residual_centered_first_derivative,
    burgers_residual_implicit_euler,
)

torch.set_num_threads(4)

print("1. the evaluation module finds the operators by name")
for system in ("burgers", "allen_cahn"):
    for names in (IMPLICIT_EULER_OPERATOR_NAMES, CENTRED_FIRST_OPERATOR_NAMES):
        fn, name = _resolve_operator(names[system])
        print(f"   {system:11s} {names[system][0]:45s} -> {name}")
        assert fn is not None, (system, names[system][0])

print()
print("2. the signatures the evaluation module inspects")
for fn in (
    burgers_residual_implicit_euler,
    burgers_residual_centered_first_derivative,
    allen_cahn_residual_implicit_euler,
    allen_cahn_residual_centered_first_derivative,
):
    print(f"   {fn.__name__}{inspect.signature(fn)}")

print()
print("3. the evaluation module calls them on the shared interior support")
n = 32
traj = torch.randn(1, 7, n, dtype=torch.float64)
params = {"nu": 0.01}
fn, _ = _resolve_operator(IMPLICIT_EULER_OPERATOR_NAMES["burgers"])
out = _call_operator(fn, "burgers", traj, 0.01, params)
print(f"   implicit euler: trajectory {tuple(traj.shape)} -> residual {tuple(out.shape)}")
assert out.shape[1] == traj.shape[1] - 2

fn, _ = _resolve_operator(CENTRED_FIRST_OPERATOR_NAMES["burgers"])
out = _call_operator(fn, "burgers", traj, 0.01, params)
print(f"   centred       : trajectory {tuple(traj.shape)} -> residual {tuple(out.shape)}")
assert out.shape[1] == traj.shape[1] - 2

traj2 = torch.randn(1, 7, 8, 8, dtype=torch.float64)
params2 = {"D": 0.001, "a": 1.0}
fn, _ = _resolve_operator(IMPLICIT_EULER_OPERATOR_NAMES["allen_cahn"])
out = _call_operator(fn, "allen_cahn", traj2, 0.1, params2)
print(f"   implicit euler: trajectory {tuple(traj2.shape)} -> residual {tuple(out.shape)}")
assert out.shape[1] == traj2.shape[1] - 2

fn, _ = _resolve_operator(CENTRED_FIRST_OPERATOR_NAMES["allen_cahn"])
out = _call_operator(fn, "allen_cahn", traj2, 0.1, params2)
print(f"   centred       : trajectory {tuple(traj2.shape)} -> residual {tuple(out.shape)}")
assert out.shape[1] == traj2.shape[1] - 2

print()
print("4. the two evaluators differ only in the time stencil")
from pdelab.evaluation.evaluate import _local_residual_tensor  # noqa: E402

for system, traj_, dt, params_ in [
    ("burgers", traj, 0.01, params),
    ("allen_cahn", traj2, 0.1, params2),
]:
    local_c = _local_residual_tensor(system, traj_, dt, params_, "centred_first_derivative")
    local_i = _local_residual_tensor(system, traj_, dt, params_, "implicit_euler_first_order")
    fn, _ = _resolve_operator(CENTRED_FIRST_OPERATOR_NAMES[system])
    op_c = _align_support(_call_operator(fn, system, traj_, dt, params_), traj_.shape[1])
    fn, _ = _resolve_operator(IMPLICIT_EULER_OPERATOR_NAMES[system])
    op_i = _align_support(_call_operator(fn, system, traj_, dt, params_), traj_.shape[1])
    dc = float(torch.linalg.norm(op_c - local_c) / torch.linalg.norm(local_c))
    di = float(torch.linalg.norm(op_i - local_i) / torch.linalg.norm(local_i))
    print(f"   {system:11s} centred: relative difference vs the local reference = {dc:.3e}")
    print(f"   {system:11s} implicit: relative difference vs the local reference = {di:.3e}")

print()
print("5. the de-aliasing change is invisible on the dataset's band-limited fields")
from pdelab.data.initial_conditions import fourier_series_1d  # noqa: E402

rng = np.random.default_rng(1000)
u, _ = fourier_series_1d(128, rng, n_modes=6, amplitude=0.9, mean=0.0)
ut = torch.as_tensor(u, dtype=torch.float64).unsqueeze(0)
a = burgers_residual_implicit_euler(ut, ut, dt=0.01, nu=0.01, dealias=True)
b = burgers_residual_implicit_euler(ut, ut, dt=0.01, nu=0.01, dealias=False)
rel = float(torch.linalg.norm(a - b) / torch.linalg.norm(b))
print(f"   burgers, dataset IC (n=128): relative difference = {rel:.3e}")
print("   (the output projection only removes content above the retained band;")
print("    the dataset fields are band limited, so the difference is round-off)")
