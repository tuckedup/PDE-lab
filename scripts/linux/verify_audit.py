"""Independent verification of the 2026-10-21 audit findings.

Each check below is a *falsification* attempt: it tries to show the
defect is present in the current source. A check that fails to
reproduce means the defect is absent (or already fixed).

Read-only and CPU-only.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

ROOT = Path("/home/atulp/projects/pdelab")
sys.path.insert(0, str(ROOT / "src"))

from pdelab.evaluation.evaluate import _pair_residuals  # noqa: E402
from pdelab.solvers.allen_cahn import AllenCahn2DSpectral  # noqa: E402
from pdelab.solvers.burgers import Burgers1DFiniteVolume  # noqa: E402
from pdelab.solvers.burgers_weno import Burgers1DWeno5  # noqa: E402

report: dict = {}


# ----------------------------------------------------------------------
# A. observation-time scheduling
# ----------------------------------------------------------------------
def check_observation_times():
    observations = np.array([0.0, 0.1, 0.2, 0.3])
    out = {}
    for name, solver, initial in [
        ("allen_cahn", AllenCahn2DSpectral(8, 8), np.zeros((8, 8))),
        ("burgers", Burgers1DWeno5(16), np.zeros(16)),
    ]:
        # A clock-valued step isolates scheduling from PDE accuracy.
        solver.step = lambda u, dt: u + dt
        solver.stable_dt = lambda *a, **k: 0.005
        saved = solver.solve_to(initial, 0.3, observations)
        actual = saved.reshape(len(saved), -1)[:, 0]
        out[name] = {
            "requested": observations.tolist(),
            "returned_clock": actual.tolist(),
            "max_abs_time_error": float(np.max(np.abs(actual - observations))),
            "reproduced_defect": bool(np.max(np.abs(actual - observations)) > 1e-9),
        }
    # The real Allen-Cahn reaction solution has a closed form, so the
    # returned frames can be converted back to the times they came from.
    s = AllenCahn2DSpectral(8, 8, D=0.0, a=1.0)
    saved = s.solve_to(np.full((8, 8), 0.5), 0.3, observations)[:, 0, 0]
    inferred = -0.5 * np.log((1 / saved**2 - 1) / (1 / 0.5**2 - 1))
    out["actual_allen_cahn_inferred_times"] = inferred.tolist()
    out["actual_allen_cahn_inferred_times_reproduced_defect"] = bool(
        np.max(np.abs(inferred - observations)) > 1e-6
    )
    return out


# ----------------------------------------------------------------------
# B. the "second-order" diagnostic computes u_tt
# ----------------------------------------------------------------------
def check_residual_derivative():
    def exact(t):
        return 1 / np.sqrt(1 + (1 / 0.2**2 - 1) * np.exp(-2 * t))

    rows = []
    for dt in [0.02, 0.01, 0.005]:
        times = np.arange(1, int(round(0.2 / dt)) + 1) * dt
        traj = np.broadcast_to(
            exact(times)[None, :, None, None], (1, len(times), 4, 4)
        ).copy()
        initial = np.full((1, 4, 4), 0.2)
        wrong = _pair_residuals(
            "allen_cahn", traj, initial, dt, {"D": 0.001, "a": 1.0}, order=2
        )
        centre = traj[:, 1:-1]
        corrected = (traj[:, 2:] - traj[:, :-2]) / (2 * dt) - (centre - centre**3)
        rows.append({
            "dt": dt,
            "existing_order2_rms": float(wrong),
            "correct_centered_first_derivative_rms": float(
                np.sqrt(np.mean(corrected**2))
            ),
        })
    # The defect: a correct first-order defect falls with dt, the
    # implemented one tends to a non-zero constant.
    defect = (
        rows[-1]["existing_order2_rms"] / rows[0]["existing_order2_rms"] > 0.9
        and rows[-1]["correct_centered_first_derivative_rms"]
        / rows[0]["correct_centered_first_derivative_rms"] < 0.2
    )
    return {"rows": rows, "reproduced_defect": bool(defect)}


# ----------------------------------------------------------------------
# C. Burgers flux splitting ignores the characteristic speed
# ----------------------------------------------------------------------
def check_flux_split():
    s = Burgers1DWeno5(32, nu=0.0)
    # The split that the implementation uses.
    f = 0.5 * s.x**2
    f_plus = np.where(f > 0.0, f, 0.0)
    f_minus = np.where(f < 0.0, f, 0.0)
    split_identically_zero = bool(np.all(f_minus == 0.0))
    f_always_nonneg = bool(np.all(f >= 0.0))

    # A decisive behavioural test: pure advection of a small
    # perturbation on a negative background must not grow.
    n = 128
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    rates = {}
    for bg in (1.0, -1.0):
        u0 = bg + 1e-4 * np.cos(2 * np.pi * 16 * x)
        sv = Burgers1DWeno5(n=n, nu=0.0)
        u = sv.solve(u0, dt=1e-4, num_steps=100)[-1]
        pert = u - bg
        rates[f"background_{bg:+.0f}"] = {
            "perturbation_energy_growth_rate": float(
                (np.sum(pert**2) / np.sum((u0 - bg) ** 2)) ** (1 / 100) - 1
            ) / 1e-4,
        }
    antidiffusion = rates["background_-1"]["perturbation_energy_growth_rate"] > 0
    return {
        "f_always_non_negative": f_always_nonneg,
        "f_minus_identically_zero": split_identically_zero,
        "advection_rates": rates,
        "negative_background_shows_antidiffusion": bool(antidiffusion),
        "reproduced_defect": bool(split_identically_zero and antidiffusion),
    }


# ----------------------------------------------------------------------
# D. FNO2D spectral coverage
# ----------------------------------------------------------------------
def check_fno2d_coverage():
    from pdelab.models.fno import SpectralConv2d

    n = 16
    conv = SpectralConv2d(1, 1, modes_x=4, modes_y=4)
    with torch.no_grad():
        # An identity operator on the retained band must reproduce its
        # input *if* that band is Hermitian symmetric (i.e. keeps the
        # negative-kx low-frequency block as well as the positive one).
        conv.weights.zero_()
        conv.weights[0, 0] = 1.0 + 0.0j
        x = torch.randn(1, 1, n, n)
        out = conv(x)
    rel = float((out - x).norm() / x.norm())

    # Which rfft2 entries does the operator actually write to?
    x_ft = torch.fft.rfft2(x)
    nonzero_in = (x_ft.abs() > 0).nonzero()
    rows = sorted({int(v) for v in nonzero_in[:, 0]})
    retained_rows = list(range(4))
    dropped_rows = [r for r in rows if r not in retained_rows]

    def kx_of(row: int) -> int:
        return row if row <= n // 2 else row - n

    dropped_kx = sorted({kx_of(r) for r in dropped_rows})

    # A single low-frequency cosine contains both +1 and -1 in kx.
    xc = torch.cos(2 * torch.pi * torch.arange(n).float() / n).reshape(1, 1, n, 1)
    xc = xc.expand(1, 1, n, n).contiguous()
    with torch.no_grad():
        outc = conv(xc)
    cosine_rel = float((outc - xc).norm() / xc.norm())

    return {
        "identity_operator_relative_error": rel,
        "identity_operator_is_exact": bool(rel < 1e-5),
        "rfft2_rows_with_energy": rows,
        "rows_written_by_operator": retained_rows,
        "rows_dropped": dropped_rows,
        "dropped_rows_correspond_to_kx": dropped_kx,
        "single_cosine_relative_error": cosine_rel,
        "reproduced_defect": bool(rel > 1e-3),
    }


# ----------------------------------------------------------------------
# E. manufactured-solution check is algebraically self-cancelling
# ----------------------------------------------------------------------
def check_manufactured_self_cancel():
    s = Burgers1DWeno5(64, nu=0.01)
    x = np.linspace(0, 1, 64, endpoint=False)
    u_ex = np.exp(-0.05) * np.cos(2 * np.pi * x)
    u_t = -u_ex
    rhs = s._rhs(u_ex)
    forcing = u_t - rhs
    dt = 1e-4
    u_next = u_ex + dt * (rhs + forcing)
    exact_next = np.exp(-(0.05 + dt)) * np.cos(2 * np.pi * x)
    err = float(np.max(np.abs(u_next - exact_next)))
    # The check passes *by construction*, without ever calling step().
    source = Path(ROOT / "src/pdelab/solvers/validation.py").read_text()
    uses_actual_step = ".step(" in source.split("manufactured")[1][:3000]
    return {
        "max_abs_error": err,
        "passes_without_calling_the_integrator": bool(err < 1e-7),
        "validation_source_calls_actual_step": bool(uses_actual_step),
        "reproduced_defect": bool(err < 1e-7),
    }


# ----------------------------------------------------------------------
# G. report selects the best test run
# ----------------------------------------------------------------------
def check_report_selection():
    src = (ROOT / "src/pdelab/reporting/report.py").read_text()
    uses_best_run = "_best_run" in src
    selects_min = 'key=lambda r: r["evaluation"]["nrmse"]["mean"]' in src
    return {
        "report_picks_a_best_run": bool(uses_best_run and selects_min),
        "reproduced_defect": bool(uses_best_run and selects_min),
    }


# ----------------------------------------------------------------------
# H. benchmark correctness uses a scalar checksum
# ----------------------------------------------------------------------
def check_benchmark_correctness():
    src = (ROOT / "src/pdelab/benchmarks/variant_runner.py").read_text()
    uses_checksum = "output_checksum" in src
    uses_tensor_compare = "allclose" in src or "max(" in src
    return {
        "uses_scalar_checksum": bool(uses_checksum),
        "uses_full_tensor_comparison": bool(uses_tensor_compare),
        "reproduced_defect": bool(uses_checksum and not uses_tensor_compare),
    }


# ----------------------------------------------------------------------
# I. lifecycle: benchmark entrypoint has no lock/preflight
# ----------------------------------------------------------------------
def check_lifecycle():
    cli = (ROOT / "src/pdelab/cli.py").read_text()
    bench_cmd = cli.split("def cmd_benchmark")[1].split("def cmd_report")[0]
    has_lock = "JobLock" in bench_cmd
    has_preflight = "preflight" in bench_cmd
    return {
        "benchmark_uses_job_lock": bool(has_lock),
        "benchmark_runs_preflight": bool(has_preflight),
        "reproduced_defect": bool(not has_lock or not has_preflight),
    }


def main() -> int:
    report["A_observation_time_scheduling"] = check_observation_times()
    report["B_residual_derivative"] = check_residual_derivative()
    report["C_flux_splitting"] = check_flux_split()
    report["D_fno2d_coverage"] = check_fno2d_coverage()
    report["E_manufactured_self_cancel"] = check_manufactured_self_cancel()
    report["G_report_selection"] = check_report_selection()
    report["H_benchmark_correctness"] = check_benchmark_correctness()
    report["I_lifecycle"] = check_lifecycle()

    out = ROOT / "campaigns" / "audit-verification" / "audit_reproduction.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    n_repro = sum(1 for v in report.values() if v.get("reproduced_defect"))
    print(f"\n{n_repro}/{len(report)} audit findings reproduced against the current source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
