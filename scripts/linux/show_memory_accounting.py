"""Print the memory/throughput accounting of a matched benchmark."""

import json
import sys
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "runs/burgers_fno1d_data_only_seed0/benchmark_matched.json"
)
d = json.loads(path.read_text())

print("workload        :", json.dumps(d.get("workload")))
print("throughput_ratio:", d.get("throughput_ratio"), "->", d.get("throughput_best_variant"))
print("memory_reduction (allocated):", d.get("memory_reduction_percent"),
      "->", d.get("memory_best_variant"))
print("memory_reduction (reserved) :", d.get("memory_reserved_reduction_percent"),
      "->", d.get("memory_reserved_best_variant"))
print("joint_pass_variant:", d.get("joint_pass_variant"))
print()
hdr = (f"{'variant':24s} {'median_s':>10s} {'peak_alloc':>11s} {'peak_resv':>10s} "
       f"{'alloc_pre':>10s} {'resv_pre':>10s} {'persist_B':>10s} {'dev_free':>9s}")
print(hdr)
print("-" * len(hdr))
for m in d.get("measurements", []):
    mem = m.get("memory", {})

    def g(k, w=10, p=5, _mem=mem):
        v = _mem.get(k)
        return "n/a".rjust(w) if v is None else f"{v:{w}.{p}g}"
    print(f"{m['variant']:24s} {m['timing']['median_seconds']:>10.5f} "
          f"{g('peak_allocated_gib', 11)} {g('peak_reserved_gib', 10)} "
          f"{g('allocated_before_reset_gib')} {g('reserved_before_reset_gib')} "
          f"{g('persistent_total_bytes', 10, 0)} {g('device_free_gib', 9, 4)}")

print()
print("per-variant joint pass:")
for v in d.get("per_variant_joint_pass", []):
    print(f"  {v['variant']:24s} tp={v['throughput_ratio']:.3f} "
          f"mem_alloc={v['memory_reduction_percent_allocated']:.2f}% "
          f"mem_resv={v['memory_reduction_percent_reserved']:.2f}% "
          f"correct={v['accuracy_within_tolerance']} "
          f"joint={v['joint_pass']}")
print()
print("caveat:", d.get("memory_caveat"))
