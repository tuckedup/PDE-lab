"""Run the full numerical validation suite and print a summary table."""

import json
import sys

from pdelab.solvers.validation import validate_allen_cahn, validate_burgers

results = []
for name, fn in [("burgers", validate_burgers), ("allen_cahn", validate_allen_cahn)]:
    r = fn()
    results.append(r)
    print(f"=== {name} ===")
    for c in r["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  {status}  {c['name']}")
        print(f"         {json.dumps(c['detail'], default=str)[:200]}")
    print(f"  => ALL PASSED: {r['passed']}  ({r['walltime_seconds']}s)")
    print()

all_passed = all(r["passed"] for r in results)
print("OVERALL:", "PASS" if all_passed else "FAIL")
sys.exit(0 if all_passed else 1)
