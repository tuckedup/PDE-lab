"""Check that all runs of a system share the same test dataset hash, and
report the run-to-run spread of the repeated configuration."""

import json
from collections import defaultdict
from pathlib import Path

RUNS = Path("/home/atulp/projects/pdelab/runs")

by_hash = defaultdict(list)
rows = []
for d in sorted(RUNS.iterdir()):
    if not d.is_dir():
        continue
    man_p = d / "manifest.json"
    ev_p = d / "evaluation.json"
    if not man_p.exists():
        continue
    man = json.loads(man_p.read_text())
    ev = json.loads(ev_p.read_text()) if ev_p.exists() else None
    dm = man.get("dataset_manifest") or {}
    test_hash = ((dm.get("files") or {}).get("test") or {}).get("sha256")
    by_hash[(man["experiment"]["system"], test_hash)].append(d.name)
    rows.append((d.name, man["experiment"]["system"], test_hash,
                 (ev or {}).get("nrmse", {}).get("mean"),
                 (man.get("training") or {}).get("best_val_loss")))

print("Test dataset hashes by system (all runs of a system must share one):")
for (system, h), names in sorted(by_hash.items()):
    print(f"  {system:12s} {h}  <- {len(names)} run(s)")
    for n in names:
        print(f"      {n}")

print("\nRepeated-configuration spread (same config, re-run, same seed):")
print(f"  {'run':46s} {'rollout nRMSE':>14s} {'best_val_loss':>15s}")
for name, _system, _h, nrmse, vl in rows:
    print(f"  {name:46s} {nrmse if nrmse is None else round(nrmse, 5):>14} "
          f"{vl if vl is None else format(vl, '.4g'):>15}")
