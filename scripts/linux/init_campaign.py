"""Create the campaign state directory and the v2 data protocol.

The v2 protocol differs from the original prototype in one scientifically
important way: the **final test set uses a fresh, disjoint set of
initial-condition seeds** that were never used during development. The
prototype test set had already been inspected while the implementation
was being written, so it cannot serve as a confirmatory test no matter
how many training seeds are run.

Usage:
    python scripts/linux/init_campaign.py
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
CAMPAIGN = ROOT / "campaigns" / "v2-recovery"

BUDGET = {
    "campaign_active_walltime_hours": 8,
    "heavy_numerical_job_elapsed_hours_total": 6,
    "max_new_training_attempts": 30,
    "tiny_cpu_fixture_elapsed_minutes_total": 30,
    "max_single_training_job_minutes": 45,
    "max_contiguous_heavy_batch_minutes": 120,
    "max_concurrent_heavy_jobs": 1,
    "max_specialist_agents": 3,
    "aggregate_numerical_cpu_threads": 4,
    "dataloader_workers": 0,
    "max_pipeline_ram_gib": 3.5,
    "max_cuda_allocated_gib": 4.5,
    "min_cuda_free_before_job_gib": 6.0,
    "min_physical_windows_free_disk_gib": 40,
    "max_new_dataset_storage_gib": 4,
    "max_new_project_storage_gib": 10,
    "max_checkpoints_per_attempt": 2,
    "max_hypothesis_rounds_per_pde": 3,
    "max_infrastructure_repair_attempts_per_failure": 2,
    "spent": {
        "heavy_numerical_job_elapsed_hours": 0.0,
        "training_attempts": 0,
        "tiny_cpu_fixture_elapsed_minutes": 0.0,
        "walltime_hours": 0.0,
    },
}


def _git(*args) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _gpu() -> dict:
    try:
        import torch

        free, total = torch.cuda.mem_get_info(0)
        return {
            "name": torch.cuda.get_device_name(0),
            "free_gib": round(free / 1024**3, 3),
            "total_gib": round(total / 1024**3, 3),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main() -> int:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    state = {
        "campaign_id": "v2-recovery",
        "created_utc": now,
        "objective": (
            "Repair the confirmed numerical/evidence defects, regenerate the "
            "data under a corrected protocol with a FRESH unseen test set, run "
            "the required 18-row main matrix, and produce an evidence-based "
            "report plus a local installable release."
        ),
        "phase": "gate2",
        "gates": {
            "gate0_inventory": "complete",
            "gate1_bug_reproduction_and_repair": "complete",
            "gate2_full_horizon_validation_and_dataset": "in_progress",
            "gate3_freeze_protocol_and_main_matrix": "pending",
            "gate4_ablations": "pending",
            "gate5_final_test_and_benchmark": "pending",
            "gate6_report_and_release": "pending",
        },
        "git": {
            "revision": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu": _gpu(),
        },
        "subagents": [
            {
                "role": "numerical/data",
                "id": "ses_e9fe6ada6ffeJS35c2uvgcmTOs",
                "status": "complete",
            },
            {
                "role": "training/evidence",
                "id": "ses_e9fe684d3ffenLTVhIoXwKpqRW",
                "status": "complete",
            },
            {
                "role": "performance/release",
                "id": "ses_e9f93337effeMQHPY11Xy0Ecvd",
                "status": "complete",
                "note": "first dispatch (ses_e9f99f0e7ffe7GS2Zmei3ogLAz) returned "
                        "an empty result and made no changes",
            },
        ],
        "last_accepted_result": (
            "All three specialist repairs integrated; 179 tests pass; the "
            "observation-time, flux-splitting, residual-derivative, FNO2D "
            "coverage, report-selection, benchmark-correctness and lifecycle "
            "defects are repaired with before/after evidence."
        ),
        "blocker": None,
    }

    (CAMPAIGN / "STATE.json").write_text(json.dumps(state, indent=2) + "\n")
    (CAMPAIGN / "BUDGET.json").write_text(json.dumps(BUDGET, indent=2) + "\n")
    (CAMPAIGN / "TASKS.json").write_text(json.dumps({"tasks": []}, indent=2) + "\n")
    (CAMPAIGN / "EVENTS.jsonl").write_text("")
    (CAMPAIGN / "NEXT_ACTION.md").write_text(
        "# Next action\n\n"
        "1. Add the fresh-test-seed split to the generator and the PDE configs.\n"
        "2. Regenerate the v2 datasets (dev train/val from the old dev seed,\n"
        "   test from a FRESH seed never used during development).\n"
        "3. Screen the lambda grid on the dev validation split only.\n"
        "4. Freeze the v2 protocol, then run the 18 required matrix rows.\n"
    )
    (CAMPAIGN / "HYPOTHESES.md").write_text(
        "# Hypotheses\n\n"
        "Each entry: falsifiable hypothesis, exact change, expected effect,\n"
        "cost, acceptance test, outcome.\n"
    )
    (CAMPAIGN / "AUDIT.md").write_text(
        "# Audit findings and dispositions\n\n"
        "See `audit-verification/audit_reproduction.json` for the independent\n"
        "reproduction of every 2026-10-21 audit finding against this checkout.\n"
    )
    (CAMPAIGN / "MATRIX.json").write_text(json.dumps({"rows": []}, indent=2) + "\n")
    (CAMPAIGN / "EVIDENCE_INDEX.json").write_text(
        json.dumps({"artifacts": []}, indent=2) + "\n"
    )
    print(f"campaign state written to {CAMPAIGN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
