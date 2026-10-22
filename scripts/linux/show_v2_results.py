"""Print the v2 per-run result table from the generated artifact."""

import json
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
d = json.loads((ROOT / "results/results.json").read_text())

hdr = (
    f"{'run_id':40s} {'rollout':>8s} {'1step':>8s} {'persist':>8s} {'numerical':>9s} "
    f"{'resid':>8s} {'refR':>8s} {'residC':>8s} {'refC':>8s}"
)
print(hdr)
print("-" * len(hdr))


def f(v, w=8, p=4):
    return "n/a".rjust(w) if v is None else f"{v:{w}.{p}g}"


for r in d["runs"]:
    rid = str(r["run_id"])
    if r.get("status") != "completed" or r.get("rollout_nrmse_mean") is None:
        continue
    if "lam0p1" in rid or "_logs" in rid:
        continue
    print(
        f"{rid:40s} {f(r['rollout_nrmse_mean'])} {f(r['one_step_nrmse_mean'])} "
        f"{f(r['persistence_nrmse_mean'])} {f(r['numerical_stepping_nrmse_mean'], 9)} "
        f"{f(r['residual_rms_implicit_euler_first_order'])} "
        f"{f(r['reference_residual_rms_implicit_euler_first_order'])} "
        f"{f(r['residual_rms_centred_first_derivative'])} "
        f"{f(r['reference_residual_rms_centred_first_derivative'])}"
    )

print()
print("Target verdicts:")
for t in d["targets"]:
    print(f"  {t['target']:34s} {t['status']:12s} measured={t.get('measured')}")
