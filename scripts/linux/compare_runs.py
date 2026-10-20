"""Print a compact comparison table over all evaluated runs."""

import json
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
RUNS = ROOT / "runs"

header = (
    f"{'run_id':44s} {'rollout':>9s} {'1step':>9s} {'persist':>8s} "
    f"{'resid':>9s} {'refR':>9s} {'residO2':>9s} {'refO2':>9s} {'params':>8s} {'train_s':>8s}"
)
print(header)
print("-" * len(header))

rows = []
for d in sorted(RUNS.iterdir()):
    if not d.is_dir():
        continue
    ev_path = d / "evaluation.json"
    man_path = d / "manifest.json"
    if not ev_path.exists() or not man_path.exists():
        continue
    ev = json.loads(ev_path.read_text())
    man = json.loads(man_path.read_text())
    if ev.get("status") != "completed":
        continue
    pc = (man.get("parameter_counts") or {}).get("total")
    tr = (man.get("training") or {}).get("walltime_seconds")
    rows.append((
        ev.get("run_id", d.name),
        ev["nrmse"]["mean"],
        ev["nrmse_one_step_teacher_forced"]["mean"],
        ev["persistence_baseline_nrmse"]["mean"],
        ev.get("residual_error_physical_rms"),
        ev.get("reference_residual_physical_rms"),
        ev.get("residual_error_physical_rms_order2"),
        ev.get("reference_residual_physical_rms_order2"),
        pc,
        tr,
    ))

for r in sorted(rows):
    def f(v, w=9, p=4):
        if v is None:
            return "n/a".rjust(w)
        return f"{v:{w}.{p}g}"
    print(
        f"{r[0]:44s} {f(r[1])} {f(r[2])} {f(r[3],8)} {f(r[4])} {f(r[5])} "
        f"{f(r[6])} {f(r[7])} {f(r[8],8,6)} {f(r[9],8,6)}"
    )

# residual improvement vs the matched data-only FNO
print()
print("Relative residual improvement vs the matched data-only FNO baseline")
print("(primary evaluator, forward-difference time derivative):")
def baseline_for(system):
    for x in rows:
        if x[0].startswith(system) and "fno" in x[0] and "data_only" in x[0]:
            return x
    return None


print()
print("Relative residual improvement of the physics-regularized FNO over the")
print("matched data-only FNO baseline (the declared target is >= 10%):")
for r in rows:
    system = r[0].split("_")[0]
    if "fno" not in r[0] or "physics" not in r[0]:
        continue
    base = baseline_for(system)
    if base is None:
        continue
    if base[4] in (None, 0) or r[4] is None:
        continue
    imp = 100.0 * (base[4] - r[4]) / base[4]
    if base[6] not in (None, 0) and r[6] is not None:
        imp2 = 100.0 * (base[6] - r[6]) / base[6]
    else:
        imp2 = float("nan")
    print(
        f"  {r[0]:42s} order1(primary)={imp:+7.2f}%   "
        f"order2(diagnostic)={imp2:+7.2f}%"
    )
