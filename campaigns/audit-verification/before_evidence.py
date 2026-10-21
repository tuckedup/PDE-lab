"""Before/after evidence for the 2026-10-21 numerical audit.

Run it against the pre-repair tree and against the repaired tree::

    PYTHONPATH=<tree>/src python campaigns/audit-verification/before_evidence.py

Every section prints the *measured* value of the property that the
repair changes, so the two runs can be compared directly.
"""

import json

import numpy as np

import pdelab
import pdelab.physics.operators as ops
from pdelab.solvers.allen_cahn import AllenCahn2DSpectral
from pdelab.solvers.burgers import Burgers1DSpectral
from pdelab.solvers.burgers_weno import Burgers1DWeno5

OUT = {}


def _section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


_section("A. observation-time scheduling")
observations = np.array([0.0, 0.1, 0.2, 0.3])
for name, solver, initial in [
    ("burgers", Burgers1DWeno5(16), np.zeros(16)),
    ("allen_cahn", AllenCahn2DSpectral(8, 8), np.zeros((8, 8))),
]:
    solver.step = lambda u, dt: u + dt
    solver.stable_dt = lambda *a, **k: 0.005
    saved = solver.solve_to(initial, 0.3, observations)
    actual = saved.reshape(len(saved), -1)[:, 0]
    err = float(np.max(np.abs(actual - observations)))
    print(f"  {name}: requested  {observations.tolist()}")
    print(f"  {name}: returned   {[round(float(v), 12) for v in actual]}")
    print(f"  {name}: max |clock - requested| = {err:.6e}   (must be < 1e-12)")
    OUT[f"A_{name}_max_abs_time_error"] = err

# the real (non-clock) path
s = Burgers1DWeno5(n=64, nu=0.01)
frames = s.solve_to(np.cos(2 * np.pi * s.x), 0.3, observations)
print(f"  burgers: n_frames returned = {frames.shape[0]} (requested {len(observations)})")
OUT["A_burgers_n_frames"] = int(frames.shape[0])
if hasattr(s, "frame_times"):
    ft = s.frame_times()
    print(f"  burgers: frame_times() = {[round(float(v), 12) for v in ft]}")
    OUT["A_burgers_frame_times_max_err"] = float(np.max(np.abs(ft - observations)))
else:
    print("  burgers: frame_times() does not exist")
    OUT["A_burgers_frame_times_max_err"] = None

_section("C. Burgers flux splitting (antidiffusion on a negative background)")
n = 128
x = np.linspace(0.0, 1.0, n, endpoint=False)
for bg in (1.0, -1.0):
    u0 = bg + 1e-4 * np.cos(2 * np.pi * 16 * x)
    sv = Burgers1DWeno5(n=n, nu=0.0)
    u = sv.solve(u0, dt=1e-4, num_steps=100)[-1]
    pert = u - bg
    rate = float((np.sum(pert**2) / np.sum((u0 - bg) ** 2)) ** (1 / 100) - 1) / 1e-4
    print(f"  background {bg:+.0f}: perturbation energy growth rate = {rate:+.6f} / unit time")
    OUT[f"C_growth_rate_background_{bg:+.0f}"] = rate

if hasattr(Burgers1DWeno5, "split_flux"):
    sv = Burgers1DWeno5(n=64)
    u_probe = np.array([1.0, -0.5, 0.2, -1.0, 0.7, 0.0])
    f_plus, f_minus, alpha = sv.split_flux(u_probe)
    resid = float(np.max(np.abs(f_plus + f_minus - 0.5 * u_probe**2)))
    print(f"  split residual |f+ + f- - f| = {resid:.3e}; alpha = {alpha}")
    print(f"  min f+' = {float(np.min(0.5 * (u_probe + alpha))):.3e} (must be >= 0)")
    print(f"  max f-' = {float(np.max(0.5 * (u_probe - alpha))):.3e} (must be <= 0)")
    OUT["C_split_residual"] = resid
else:
    print("  split_flux() does not exist")
    OUT["C_split_residual"] = None

_section("F. de-aliasing")
for n_mask in (32, 64, 128):
    mask = Burgers1DSpectral._dealias_mask(n_mask)
    mirror = mask[(-np.arange(n_mask)) % n_mask]
    hermitian = bool(np.array_equal(mask, mirror))
    print(f"  Burgers1DSpectral mask n={n_mask:4d}: hermitian_symmetric={hermitian} "
          f"(kept {int(mask.sum())} modes)")
    OUT[f"F_burgers_mask_hermitian_n{n_mask}"] = hermitian
    if hasattr(AllenCahn2DSpectral, "_dealias_mask"):
        mask2 = AllenCahn2DSpectral._dealias_mask(n_mask)
        mirror2 = mask2[(-np.arange(n_mask)) % n_mask]
        herm2 = bool(np.array_equal(mask2, mirror2))
        print(f"  AllenCahn2DSpectral mask n={n_mask:4d}: hermitian_symmetric={herm2} "
              f"(kept {int(mask2.sum())} modes)")
        OUT[f"F_ac_mask_hermitian_n{n_mask}"] = herm2

def _padded_product_1d(u, power, keep):
    """Alias-free retained-band product on a 2x padded grid (self-contained)."""
    u = np.asarray(u, dtype=np.float64)
    n = u.size
    u_hat = np.fft.fft(u)
    modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
    sel = np.abs(modes) <= keep
    big_index = np.where(modes >= 0, modes, modes + 2 * n)
    padded_hat = np.zeros(2 * n, dtype=complex)
    padded_hat[big_index[sel]] = 2.0 * u_hat[sel]
    padded = np.fft.ifft(padded_hat).real
    product_hat = np.fft.fft(padded**power)
    out = np.zeros(n, dtype=complex)
    out[sel] = product_hat[big_index[sel]] / 2.0
    return np.fft.ifft(out).real


# residual aliasing error against an independently padded product
try:
    n_a = 128
    rng = np.random.default_rng(31337)
    u = rng.standard_normal(n_a)
    spec = Burgers1DSpectral(n=n_a, nu=0.01)
    keep = n_a // 3
    exact = 0.5 * _padded_product_1d(u, 2, keep)
    masked = np.fft.ifft(
        np.fft.fft(0.5 * np.fft.ifft(np.fft.fft(u) * spec._mask).real**2) * spec._mask
    ).real
    rel = float(np.linalg.norm(masked - exact) / np.linalg.norm(exact))
    print(f"  Burgers spectral nonlinear product: relative aliasing error = {rel:.3e}")
    OUT["F_burgers_aliasing_error"] = rel
except Exception as exc:  # pragma: no cover
    print(f"  could not evaluate the aliasing error: {exc}")
    OUT["F_burgers_aliasing_error"] = None

_section("E. manufactured solution (is the integrator actually called?)")
try:
    from pdelab.solvers.validation import validate_allen_cahn, validate_burgers

    for title, fn in [("burgers", validate_burgers), ("allen_cahn", validate_allen_cahn)]:
        result = fn()
        check = next(c for c in result["checks"] if "manufactured_solution" in c["name"])
        detail = check["detail"]
        print(f"  {title}: check '{check['name']}' passed={check['passed']}")
        print(f"      max_abs_error       = {detail.get('max_abs_error')}")
        print(f"      driven_through      = {detail.get('driven_through')}")
        print(f"      integrator_calls    = "
              f"{[r.get('integrator_calls') for r in detail.get('rows', [])]}")
        OUT[f"E_{title}_mms_max_abs_error"] = detail.get("max_abs_error")
        OUT[f"E_{title}_mms_integrator_calls"] = [
            r.get("integrator_calls") for r in detail.get("rows", [])
        ]
except Exception as exc:  # pragma: no cover
    print(f"  could not evaluate the manufactured-solution checks: {exc}")

_section("B. the diagnostic time derivative operators")
for name in [
    "time_derivative_centered_first_order",
    "time_derivative_forward_first_order",
    "burgers_residual_implicit_euler",
    "burgers_residual_centered_first_derivative",
    "allen_cahn_residual_implicit_euler",
    "allen_cahn_residual_centered_first_derivative",
]:
    print(f"  operators.{name}: {'present' if hasattr(ops, name) else 'MISSING'}")
    OUT[f"B_{name}"] = hasattr(ops, name)

_section("dataset manifest")
try:
    import tempfile

    from pdelab.config import PDESystemConfig
    from pdelab.data.generate import DatasetGenerator

    system = PDESystemConfig(
        name="burgers", dimension=1, solver="finite_volume_weno5_js",
        solver_grid=(64,), observation_grid=(32,),
        observation_times=[0.0, 0.01, 0.02, 0.03], t_final=0.03,
        params={"nu": 0.01},
        initial_condition={"kind": "fourier", "n_modes": 4, "amplitude": 0.9, "mean": 0.0},
        reference_grid=(64,),
    )
    with tempfile.TemporaryDirectory() as tmp:
        gen = DatasetGenerator(system=system, out_dir=tmp, n_train=1, n_val=1, n_test=1,
                               seed=99, profile="evidence")
        manifest = gen.generate()
    rec = manifest["trajectories"][0]
    print(f"  manifest trajectory keys: {sorted(rec.keys())}")
    print(f"  'actual_observation_times' present: "
          f"{'actual_observation_times' in rec}")
    print(f"  manifest 'observation_scheduler' present: "
          f"{'observation_scheduler' in manifest}")
    print(f"  manifest 'time_stepping_policy' present: "
          f"{'time_stepping_policy' in manifest}")
    OUT["manifest_actual_observation_times"] = "actual_observation_times" in rec
except Exception as exc:  # pragma: no cover
    print(f"  could not generate a dataset: {exc}")

print()
print("=" * 78)
print("pdelab source tree:", pdelab.__file__)
print(json.dumps(OUT, indent=2, default=str))
