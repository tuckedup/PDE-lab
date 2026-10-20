"""Summarize a benchmark artifact."""

import json
import sys
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")

path = Path(sys.argv[1])
if not path.is_absolute():
    path = ROOT / "runs" / sys.argv[1] / "benchmark_matched.json"
d = json.loads(path.read_text())

print("run_id            :", d.get("run_id"))
print("profile           :", d.get("profile"))
print("isolation         :", d.get("isolation"))
print("device            :", (d.get("device") or {}).get("name"))
print("workload          :", json.dumps(d.get("workload")))
print("throughput_ratio  :", d.get("throughput_ratio"), "->", d.get("throughput_best_variant"))
print("memory_reduction  :", d.get("memory_reduction_percent"), "->", d.get("memory_best_variant"))
print()
for m in d.get("measurements", []):
    t = m.get("timing", {})
    mem = m.get("memory", {})
    print(
        f"  bs={m.get('batch_size')} {m.get('variant'):24s} "
        f"status={m.get('status', 'ok'):8s} "
        f"med={t.get('median_seconds', float('nan')):.5f}s "
        f"min={t.get('min_seconds', float('nan')):.5f}s "
        f"max={t.get('max_seconds', float('nan')):.5f}s "
        f"peak_alloc={mem.get('peak_allocated_gib', float('nan')):.5f}GiB "
        f"compile={m.get('compile_seconds', 0.0):.1f}s "
        f"acc_ok={m.get('accuracy_within_tolerance')}"
    )
    if m.get("status") == "failed":
        print("      error:", m.get("error", "")[-400:])
