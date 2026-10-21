"""Smoke test of the repaired inference benchmark (GPU, a few seconds).

This is **not** the full benchmark: one block of one iteration per
variant, on a copy of an existing run directory, so that the existing
artifact under ``runs/`` is not overwritten.

It proves the whole path: the isolated variant processes, the full
output tensors, the element-wise accuracy comparison, the content
identity and the artifact write.

Usage::

    .venv/bin/python campaigns/performance/benchmark_smoke.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from pdelab.benchmarks.inference import run_benchmark
from pdelab.config import project_root

SOURCE_RUN = "burgers_fno1d_data_only_seed0"
SMOKE_DIR = Path("/tmp/pdelab_perf/bench_runs/smoke")


def main() -> int:
    src = project_root() / "runs" / SOURCE_RUN
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True)
    for name in ("best.pt", "manifest.json"):
        shutil.copy2(src / name, SMOKE_DIR / name)

    result = run_benchmark(
        SMOKE_DIR,
        profile="matched",
        device="cuda",
        n_blocks=1,
        iters_per_block=1,
        correctness_batches=2,
        correctness_batch_size=1,
        preflight_report={"timestamp_utc": "smoke", "ok": True, "violations": [],
                          "warnings": []},
    )
    print("status:", result["status"])
    print("artifact:", result.get("benchmark_path"))
    print()
    print("--- accuracy section ---")
    print(json.dumps(result["accuracy"], indent=2)[:4000])
    print()
    print("--- identity ---")
    identity = result["identity"]
    print("checkpoint_sha256       :", identity["checkpoint_sha256"])
    print("evaluator.sha256        :", identity["evaluator"]["sha256"])
    print("evaluator.id            :", identity["evaluator"]["id"])
    print("protocol.id             :", identity["protocol"]["id"])
    print("protocol.benchmark.id   :", identity["protocol"]["benchmark"]["id"])
    print("resolved_settings_sha256:", identity["resolved_settings_sha256"])
    print("dataset.split           :", identity["dataset"]["split"])
    print("dataset.sha256          :", identity["dataset"]["sha256"])
    print("normalization.sha256    :", identity["normalization"]["sha256"])
    print()
    print("--- per-variant verdict ---")
    for m in result["measurements"]:
        acc = m.get("accuracy", {})
        print(
            f"{m['variant']:24s} within_tolerance={m.get('accuracy_within_tolerance')} "
            f"max_abs_error={acc.get('max_abs_error')} "
            f"rel_l2={acc.get('relative_l2')} nrmse_delta={acc.get('nrmse_delta')}"
        )
    print()
    print("throughput_ratio:", result.get("throughput_ratio"))
    print("memory_reduction_percent:", result.get("memory_reduction_percent"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
