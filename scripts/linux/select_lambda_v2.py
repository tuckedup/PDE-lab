"""The v2 lambda-selection rule (Arm A and Arm B), on the DEV split only.

Arm A - the required physics comparison
    Among the predeclared non-zero candidates [0.01, 0.1], select the
    candidate whose mean *validation* free-rollout nRMSE is within 5% of
    the best non-zero candidate, breaking ties by the lowest mean
    *validation* primary residual and then by the smaller lambda.

Arm B - the overall method recommendation
    Compare Arm A's selected non-zero candidate against lambda = 0 under
    the same global 5% accuracy constraint and the same residual
    tie-break, and report the pure-accuracy winner as well.

The TEST split is never read here. The rule was fixed prospectively for
v2 (see configs/protocol_v2.yaml); its provenance - that the 5%
constraint was first written down while looking at prototype results -
is recorded honestly in that file.

Selection uses the mean over the seeds for which *every* candidate is
available, so the comparison is over the same training runs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
RUNS = ROOT / "runs"
OUT = ROOT / "configs" / "lambda_selection.json"
PY = str(ROOT / ".venv/bin/python")

SYSTEMS = ["burgers", "allen_cahn"]
SEEDS = [0, 1, 2]
NONZERO = [0.01, 0.1]
ACCURACY_TOLERANCE = 0.05


def evaluate_on_val(run_id: str) -> dict | None:
    """Validation-split evaluation for a run (cached, identity-checked)."""
    out = RUNS / run_id / "evaluation_val.json"
    if not (RUNS / run_id / "manifest.json").exists():
        return None
    if out.exists():
        try:
            return json.loads(out.read_text())
        except (ValueError, OSError):
            return None
    proc = subprocess.run(
        [PY, "-m", "pdelab.cli", "evaluate", "--run-id", run_id, "--split", "val"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if proc.returncode != 0 or not out.exists():
        print(f"    evaluation of {run_id} failed: {proc.stderr[-400:]}")
        return None
    return json.loads(out.read_text())


def residual_of(ev: dict) -> float | None:
    """The declared primary (implicit-Euler first-order) validation residual."""
    for key in (
        "residual_rms_implicit_euler_first_order",
        "residual_error_physical_rms",
        "residual_physical_rms",
    ):
        if ev.get(key) is not None:
            return float(ev[key])
    return None


def main() -> int:
    selection: dict[str, dict] = {}
    for system in SYSTEMS:
        fno = "fno1d" if system == "burgers" else "fno2d"
        run_ids = {
            0.0: [f"{system}_{fno}_data_only_seed{s}" for s in SEEDS],
            0.01: [f"{system}_{fno}_physics_seed{s}" for s in SEEDS],
            0.1: [f"{system}_{fno}_physics_lam0p1_seed{s}" for s in SEEDS],
        }
        per_lambda: dict[float, list[dict]] = {}
        print(f"{system}:")
        for lam, ids in run_ids.items():
            rows = []
            for run_id in ids:
                ev = evaluate_on_val(run_id)
                if ev is None:
                    continue
                res = residual_of(ev)
                rows.append({
                    "run_id": run_id,
                    "seed": ev.get("seed"),
                    "val_rollout_nrmse": ev["nrmse"]["mean"],
                    "val_one_step_nrmse": ev["nrmse_one_step_teacher_forced"]["mean"],
                    "val_primary_residual": res,
                })
            if rows:
                per_lambda[lam] = rows
                r = [x["val_rollout_nrmse"] for x in rows]
                res = [x["val_primary_residual"] for x in rows if x["val_primary_residual"] is not None]
                print(f"  lambda={lam:<5} seeds={len(rows)} "
                      f"mean_val_rollout={sum(r)/len(r):.6g} "
                      f"({', '.join(f'{v:.4g}' for v in r)})"
                      + (f"  mean_val_resid={sum(res)/len(res):.6g}" if res else ""))
            else:
                print(f"  lambda={lam:<5} no runs available yet")

        if not per_lambda:
            continue

        # Only compare lambdas measured on the same set of seeds.
        seed_sets = [set(x["seed"] for x in rows) for rows in per_lambda.values()]
        common = set.intersection(*seed_sets) if seed_sets else set()
        candidates = []
        for lam, rows in per_lambda.items():
            sel = [x for x in rows if x["seed"] in common]
            if not sel:
                continue
            res = [x["val_primary_residual"] for x in sel
                   if x["val_primary_residual"] is not None]
            candidates.append({
                "lambda": lam,
                "n_seeds": len(sel),
                "seeds": sorted(x["seed"] for x in sel),
                "mean_val_rollout_nrmse": sum(x["val_rollout_nrmse"] for x in sel) / len(sel),
                "mean_val_one_step_nrmse": sum(x["val_one_step_nrmse"] for x in sel) / len(sel),
                "mean_val_primary_residual": (sum(res) / len(res)) if res else None,
                "per_seed": sel,
            })
        if not candidates:
            continue

        def admissible_within(cands: list[dict], tol: float) -> list[dict]:
            best = min(cands, key=lambda c: (c["mean_val_rollout_nrmse"], c["lambda"]))
            return [
                c for c in cands
                if c["mean_val_rollout_nrmse"] <= best["mean_val_rollout_nrmse"] * (1 + tol)
            ]

        # ---- Arm A: the required physics comparison (non-zero candidates) ----
        nonzero = [c for c in candidates if c["lambda"] in NONZERO]
        if not nonzero:
            print("  no non-zero candidate available yet; Arm A deferred")
            continue
        adm_a = admissible_within(nonzero, ACCURACY_TOLERANCE)
        arm_a = min(
            adm_a,
            key=lambda c: (
                c["mean_val_primary_residual"]
                if c["mean_val_primary_residual"] is not None
                else float("inf"),
                c["lambda"],
            ),
        )
        arm_a_pure_accuracy = min(nonzero, key=lambda c: (c["mean_val_rollout_nrmse"], c["lambda"]))

        # ---- Arm B: the overall recommendation (non-zero vs zero) ----
        adm_b = admissible_within(candidates, ACCURACY_TOLERANCE)
        arm_b = min(
            adm_b,
            key=lambda c: (
                c["mean_val_primary_residual"]
                if c["mean_val_primary_residual"] is not None
                else float("inf"),
                c["lambda"],
            ),
        )
        arm_b_pure_accuracy = min(candidates, key=lambda c: (c["mean_val_rollout_nrmse"], c["lambda"]))
        zero = next((c for c in candidates if c["lambda"] == 0.0), None)
        arm_a_satisfies_global = (
            zero is not None
            and arm_a["mean_val_rollout_nrmse"]
            <= zero["mean_val_rollout_nrmse"] * (1 + ACCURACY_TOLERANCE)
        )

        selection[system] = {
            "arm_a_selected_lambda": arm_a["lambda"],
            "arm_a_selected_mean_val_rollout_nrmse": arm_a["mean_val_rollout_nrmse"],
            "arm_a_selected_mean_val_primary_residual": arm_a["mean_val_primary_residual"],
            "arm_a_pure_accuracy_lambda": arm_a_pure_accuracy["lambda"],
            "arm_a_admissible_lambdas": [c["lambda"] for c in adm_a],
            "arm_b_overall_recommended_lambda": arm_b["lambda"],
            "arm_b_pure_accuracy_lambda": arm_b_pure_accuracy["lambda"],
            "arm_b_admissible_lambdas": [c["lambda"] for c in adm_b],
            "arm_a_satisfies_global_accuracy_constraint": bool(arm_a_satisfies_global),
            "accuracy_tolerance": ACCURACY_TOLERANCE,
            "seeds_used": arm_a["seeds"],
            "candidates": candidates,
            "selection_split": "validation (development)",
            "test_split_used": False,
            "rule": (
                "Arm A: among the non-zero candidates [0.01, 0.1] within 5% of "
                "the best non-zero mean validation rollout nRMSE, take the "
                "lowest mean validation primary residual, then the smaller "
                "lambda. Arm B: the same rule applied over {0, 0.01, 0.1}."
            ),
        }
        print(f"  -> Arm A: lambda={arm_a['lambda']} "
              f"(rollout {arm_a['mean_val_rollout_nrmse']:.6g}, "
              f"resid {arm_a['mean_val_primary_residual']})  "
              f"pure-accuracy would be {arm_a_pure_accuracy['lambda']}")
        print(f"  -> Arm B: lambda={arm_b['lambda']}  "
              f"pure-accuracy would be {arm_b_pure_accuracy['lambda']}  "
              f"non-zero arm satisfies the global 5% constraint: {arm_a_satisfies_global}")
        print(f"     admissible: A={[c['lambda'] for c in adm_a]} "
              f"B={[c['lambda'] for c in adm_b]}\n")

    OUT.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
