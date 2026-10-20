"""Summarise the sweep dry-run plan."""

import json
import subprocess
import sys

cmd = [
    ".venv/bin/python", "-m", "pdelab.cli", "sweep",
    "--config", "configs/experiments/main_matrix.yaml",
    "--profile", "full", "--dry-run",
]
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.returncode != 0:
    print(proc.stdout)
    print(proc.stderr)
    sys.exit(1)

plan = json.loads(proc.stdout)
print(f"profile                : {plan['profile']}")
print(f"total runs             : {plan['n_runs']}")
print(f"  required (minimum)   : {plan['n_required_runs']}  "
      f"({plan['required_minimum_matrix']})")
print(f"  optional extension   : {plan['n_optional_runs']}")
print(f"estimated total minutes: {plan['estimated_total_minutes']}")
print(f"estimated required min : {plan['estimated_required_minutes']}")
print()
missing = 0
for r in plan["runs"]:
    flag = " (optional)" if r.get("optional") else ""
    print(f"  {r['name']:44s} epochs={r['epochs']:3d}{flag}")
