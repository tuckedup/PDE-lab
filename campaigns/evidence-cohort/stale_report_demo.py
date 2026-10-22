"""Demonstration of the evidence states on a deliberately damaged artifact set.

The script

1. assesses every *real* run in ``runs/`` and prints its evidence state
   with every reason (this shows the staleness that is present in the
   repository today), and
2. builds a deliberately damaged copy of a run - a truncated artifact, an
   artifact that is older than the checkpoint it evaluates, an artifact
   written by a different evaluator - and shows that the report returns
   a *state with a reason* and no number for each of them.

The result is written to ``campaigns/evidence-cohort/stale_report_demo.json``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pdelab.evaluation.metrics import (
    canonical_json_sha256,
    evaluator_identity,
    sha256_file,
    write_json_atomic,
)
from pdelab.reporting.report import (
    assess_run_evidence,
    build_report,
    collect_runs,
)

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "stale_report_demo.json"


def assess_real_runs() -> list[dict]:
    rows = []
    runs = collect_runs(RUNS)
    for run in runs:
        evidence = assess_run_evidence(run, cohort={}, evaluator_now=evaluator_identity())
        if evidence["state"] == "valid":
            continue
        rows.append({
            "run_id": run["run_id"],
            "state": evidence["state"],
            "reasons": evidence["reasons"],
            "evaluation_mtime": (run["dir"] / "evaluation.json").stat().st_mtime
            if (run["dir"] / "evaluation.json").exists()
            else None,
            "checkpoint_mtime": (run["dir"] / "best.pt").stat().st_mtime
            if (run["dir"] / "best.pt").exists()
            else None,
        })
    return rows


def _copy_run(root: Path, destination_id: str, source_id: str) -> tuple[Path, Path]:
    src = RUNS / source_id
    dst = root / "runs" / destination_id
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "evaluation.json", "best.pt"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    return dst, src


def _identity(run_dir: Path, dataset: Path) -> dict:
    return {
        "evaluator": evaluator_identity(),
        "protocol": {"id": "pdelab-free-rollout/2", "sha256": "proto"},
        "resolved_settings": {"split": "test"},
        "resolved_settings_sha256": canonical_json_sha256({"split": "test"}),
        "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "dataset": {"split": "test", "path": str(dataset), "sha256": sha256_file(dataset)},
        "normalization": {"mean": 0.0, "std": 1.0},
    }


def build_damaged_set(root: Path) -> dict:
    """A runs directory with three deliberately damaged runs."""
    source_id = "burgers_fno1d_data_only_seed0"
    runs_root = root / "runs"
    results = {}

    # (1) a truncated artifact -> "failed"
    run_dir, _ = _copy_run(root, "truncated", source_id)
    (run_dir / "evaluation.json").write_text('{"status": "completed", "nrmse"')

    # (2) an artifact that is older than the checkpoint it evaluates -> "stale"
    run_dir, src = _copy_run(root, "stale", source_id)
    artifact = json.loads((run_dir / "evaluation.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    split = Path(manifest["experiment"]["dataset_dir"]) / "burgers_test.h5"
    artifact["identity"] = _identity(run_dir, split)
    write_json_atomic(run_dir / "evaluation.json", artifact)
    later = time.time() + 120
    os.utime(run_dir / "best.pt", (later, later))

    # (3) an artifact written by a different evaluator -> "invalid_protocol"
    run_dir, _ = _copy_run(root, "foreign", source_id)
    artifact = json.loads((run_dir / "evaluation.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    split = Path(manifest["experiment"]["dataset_dir"]) / "burgers_test.h5"
    identity = _identity(run_dir, split)
    identity["evaluator"] = {
        "id": "pdelab-evaluation/1",
        "sha256": "0" * 64,
        "files": {"src/pdelab/evaluation/evaluate.py": "0" * 64},
    }
    artifact["identity"] = identity
    write_json_atomic(run_dir / "evaluation.json", artifact)
    results["runs_root"] = str(runs_root)
    return results


def main() -> int:
    out: dict = {"real_runs": assess_real_runs()}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_damaged_set(root)
        cohort_path = root / "cohort.yaml"
        cohort_path.write_text(
            "schema: pdelab-evidence-cohort/1\n"
            "cohort_id: damaged-set\n"
            "evaluator:\n"
            "  id: pdelab-evaluation/2\n"
            f"  sha256: {evaluator_identity()['sha256']}\n"
            "protocol:\n"
            "  id: pdelab-free-rollout/2\n"
            "members:\n"
            "  - id: truncated\n"
            "    system: burgers\n"
            "    architecture: fno1d\n"
            "    condition: data_only\n"
            "    seeds: [0]\n"
            "    run_ids: [truncated]\n"
            "  - id: stale\n"
            "    system: burgers\n"
            "    architecture: fno1d\n"
            "    condition: data_only\n"
            "    seeds: [0]\n"
            "    run_ids: [stale]\n"
            "  - id: foreign\n"
            "    system: burgers\n"
            "    architecture: fno1d\n"
            "    condition: data_only\n"
            "    seeds: [0]\n"
            "    run_ids: [foreign]\n"
            "targets:\n"
            "  - name: burgers_nrmse\n"
            "    metric: nrmse\n"
            "    members: [truncated, stale, foreign]\n"
        )
        report = build_report(
            output_dir=root / "results",
            make_plots=False,
            runs_dir=root / "runs",
            cohort_path=cohort_path,
            write_docs=False,
        )
        payload = json.loads((root / "results" / "results.json").read_text())
        out["damaged_set"] = {
            "targets": payload["targets"],
            "markdown_excerpt": (
                (root / "results" / "RESULTS.md").read_text().split("## Runs")[0]
            ),
            "targets_met": report["targets_met"],
        }

    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["damaged_set"]["targets"], indent=2))
    print()
    print(f"real runs assessed: {len(out['real_runs'])}")
    for row in out["real_runs"][:6]:
        print(f"   {row['run_id']}: {row['state']}")
        for reason in row["reasons"]:
            print(f"       - {reason}")
    print(f"... written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
