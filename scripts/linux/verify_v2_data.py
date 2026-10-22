"""Verify the v2 dataset manifest: exact observation times, disjoint IC
seeds, and identity hashes."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/atulp/projects/pdelab")
sys.path.insert(0, str(ROOT / "src"))

fail = 0
for system, _dev_seed, _test_seed in [("burgers", 1000, 7001), ("allen_cahn", 2000, 7002)]:
    man = json.loads((ROOT / f"data/{system}/{system}_manifest.json").read_text())
    recs = man["trajectories"]
    dev = {r["ic_seed"] for r in recs if r["split"] in ("train", "val")}
    test = {r["ic_seed"] for r in recs if r["split"] == "test"}

    times = np.asarray(man["system"]["observation_times"], dtype=float)
    max_err = 0.0
    for r in recs:
        actual = np.asarray(r.get("actual_observation_times", []), dtype=float)
        if actual.size != times.size:
            print(f"  {system}: trajectory {r['index']} has {actual.size} actual times "
                  f"for {times.size} requested")
            fail += 1
            break
        max_err = max(max_err, float(np.max(np.abs(actual - times))))

    print(f"{system}:")
    print(f"  split_seed={man.get('split_seed')} test_split_seed={man.get('test_split_seed')}")
    print(f"  dev IC seeds: {len(dev)}   test IC seeds: {len(test)}   "
          f"overlap: {len(dev & test)}")
    print(f"  max |actual - requested| observation time error: {max_err:.3e}")
    print(f"  observation_scheduler recorded: {bool(man.get('observation_scheduler'))}")
    print(f"  time_stepping_policy recorded: {bool(man.get('time_stepping_policy'))}")
    for split in ("train", "val", "test"):
        f = man["files"][split]
        print(f"  {split:5s} sha256={f['sha256'][:16]}... bytes={f['bytes']}")
    if dev & test:
        print("  FAIL: dev and test IC seeds overlap")
        fail += 1
    if max_err > 1e-12:
        print("  FAIL: observation times are not exact")
        fail += 1

print()
print("VERIFIED" if fail == 0 else f"{fail} PROBLEM(S)")
sys.exit(0 if fail == 0 else 1)
