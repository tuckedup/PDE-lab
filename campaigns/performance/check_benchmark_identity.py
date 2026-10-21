"""Check that the report can bind the benchmark artifact.

The report (``src/pdelab/reporting/report.py``) decides whether an
artifact may be used by looking at the *content identity* it records.
This script runs the report's own evidence assessment on a benchmark
artifact and prints the state and the reasons, so that the field names
of the identity are verified against the code that reads them.

Usage::

    .venv/bin/python campaigns/performance/check_benchmark_identity.py \
        /tmp/pdelab_perf/bench_smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

from pdelab.config import load_yaml, project_root
from pdelab.evaluation.metrics import evaluator_identity
from pdelab.reporting.report import (
    DEFAULT_COHORT_PATH,
    assess_run_evidence,
    collect_runs,
)


def main() -> int:
    run_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/tmp/pdelab_perf/bench_runs"
    )
    runs = collect_runs(run_root)
    if not runs:
        print(f"no run directory at {run_root}")
        return 1
    cohort = load_yaml(project_root() / DEFAULT_COHORT_PATH)
    evaluator_now = evaluator_identity()

    for run in runs:
        for artifact in ("evaluation", "benchmark"):
            evidence = assess_run_evidence(run, cohort, evaluator_now, artifact=artifact)
            print(f"--- {artifact} ---")
            print("state :", evidence["state"])
            print("reason:", evidence["reason"])
            for r in evidence.get("reasons", []):
                print("   *", r)
        payload = run.get("benchmark") or {}
        identity = payload.get("identity") or {}
        print()
        print("identity field names the report reads:")
        print("  identity.evaluator.sha256        :", identity.get("evaluator", {}).get("sha256"))
        print("  identity.evaluator.id            :", identity.get("evaluator", {}).get("id"))
        print("  identity.protocol.id              :", identity.get("protocol", {}).get("id"))
        print("  identity.resolved_settings_sha256 :", identity.get("resolved_settings_sha256"))
        print("  identity.checkpoint_sha256        :", identity.get("checkpoint_sha256"))
        print("  identity.dataset.split            :", identity.get("dataset", {}).get("split"))
        print("  identity.dataset.path             :", identity.get("dataset", {}).get("path"))
        print("  identity.dataset.sha256           :", identity.get("dataset", {}).get("sha256"))
        print("  identity.normalization.sha256     :", identity.get("normalization", {}).get("sha256"))
        print("  payload.system / payload.split    :", payload.get("system"), "/", payload.get("split"))
        sidecar = run_root / run["run_id"] / "benchmark_matched.json.sha256"
        print("  digest sidecar                    :", sidecar.exists())
    return 0


if __name__ == "__main__":
    sys.exit(main())
