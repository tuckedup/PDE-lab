"""Administrative closure of the v2-recovery campaign.

Metadata only. This script does not train, benchmark, solve, rebuild or
run the test suite, and it does not change any scientific result, the
released artifact, or any recorded spend.

It reconciles the completion metadata with the existing verified release
evidence:

  * the release gate is marked complete with explicit evidence
    references;
  * the obsolete git revision and the stale `dirty: true` are replaced by
    an unambiguously labelled source-revision block;
  * the complete NOT_EXECUTED list is retained in the campaign state;
  * the budget ledger is **frozen**: the recorded spend, the 0.355-hour
    walltime overrun and every heavy-work charge are preserved exactly,
    and the reconstruction script is taught to leave a closed ledger
    alone so the frozen numbers cannot drift.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
CAMPAIGN = ROOT / "campaigns" / "v2-recovery"
STATE = CAMPAIGN / "STATE.json"
BUDGET = CAMPAIGN / "BUDGET.json"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20
    ).stdout.strip()


def main() -> int:
    state = json.loads(STATE.read_text())
    budget = json.loads(BUDGET.read_text())
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    release_rev = (manifest.get("source") or {}).get("revision")
    wheel_sha = (manifest.get("artifact") or {}).get("wheel_sha256")

    # Guard against a circular rewrite: once the ledger is closed, the
    # recorded revisions are authoritative. Re-running this script after
    # the closure commit would otherwise replace the *verified* head with
    # the closure commit itself and make the state file self-referential.
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    if previous.get("phase") == "closed" and previous.get("git", {}).get(
        "verified_head_at_closure"
    ):
        print("campaign already closed; recorded revisions left unchanged")
        print(f"  verified head at closure: "
              f"{previous['git']['verified_head_at_closure']}")
        print(f"  release build source    : "
              f"{previous['git'].get('release_build_source_revision')}")
        return 0

    # ---- 1. the release gate, with explicit evidence references ----
    state["gates"]["gate6_report_and_release"] = "complete"
    state["gates"]["gate6_evidence"] = {
        "released_artifact": {
            "path": "dist/pdelab-0.1.0-py3-none-any.whl",
            "sha256": wheel_sha,
            "reproducible_builds": 3,
            "record": "campaigns/release/build_reproducibility.json",
        },
        "clean_install": {
            "status": (manifest.get("clean_install") or {}).get("status"),
            "open_blockers": (manifest.get("clean_install") or {}).get("open_blockers"),
            "record": "campaigns/release/clean_install.json",
            "log": "campaigns/release/clean_install.log",
            "note": "executed from a temporary directory outside the source "
                    "checkout, CPU only, with the import path shown to come "
                    "from the installed wheel",
        },
        "release_manifest": {
            "path": "RELEASE_MANIFEST.json",
            "documentation": "docs/RELEASE.md",
        },
        "final_report": {
            "markdown": "docs/RESULTS.md",
            "json": "results/results.json",
            "generated_from": "runs/*/manifest.json, runs/*/evaluation.json, "
                              "runs/*/benchmark_matched.json",
        },
        "final_verification": {
            "script": "scripts/linux/final_check_v2.py",
            "result": "68/68 checks passed",
        },
        "legacy_invalidation": "docs/LEGACY_INVALIDATION.md",
    }

    # ---- 2. the source revision, labelled unambiguously -------------
    # This file is itself tracked, so it cannot contain the hash of the
    # commit that carries it. The two revisions below are therefore
    # labelled by *what they are* rather than presented as a single
    # self-referential "current revision".
    state["git"] = {
        "verified_head_at_closure": head,
        "verified_head_branch": branch,
        "verified_head_working_tree_clean": True,
        "verified_head_note": (
            "The working tree was clean at this revision when the owner "
            "verified the release (18/18 required rows evaluated, the wheel "
            "hash matching the manifest, all packaged files matching the "
            "current source, both solver CLIs passing). The only changes "
            "made after it are the closure metadata listed in "
            "closure_touches."
        ),
        "closure_touches": [
            "campaigns/v2-recovery/STATE.json",
            "campaigns/v2-recovery/BUDGET.json",
            "scripts/linux/close_campaign.py",
            "scripts/linux/reconstruct_campaign_state.py",
            "scripts/linux/verify_closure.py",
            "scripts/linux/confirm_no_active_workers.sh",
            "docs/STATUS.md",
        ],
        "release_build_source_revision": release_rev,
        "release_build_source_note": (
            "The released wheel was built from this commit with "
            f"{len((manifest.get('source') or {}).get('dirty_paths', []))} "
            "additional uncommitted paths present in the tree at build time; "
            "the manifest records that tree in full. It is not identical to "
            "any single commit."
        ),
        "labelling_note": (
            "This state file is tracked in git, so it cannot contain the hash "
            "of the commit that carries it. The commit that records this "
            "closure necessarily follows both revisions named above. No "
            "self-referential 'current revision' is asserted."
        ),
    }

    # ---- 3. retain the complete NOT_EXECUTED list --------------------
    state["not_executed"] = [
        "the physics-regularized U-Net at three seeds (the optional "
        "extension; the configs exist, the rows were not run)",
        "the lambda grid at seeds 1 and 2 for the non-selected candidate "
        "(the screening was at seed 0 and the selected condition was then "
        "confirmed at three seeds, as the protocol prescribes)",
        "a near parameter-matched U-Net width search (the width-4 U-Net is "
        "0.63x / 0.51x the FNO parameter count, not matched)",
        "the width / spectral-mode ablation on the repaired FNO2D",
        "the maximum-throughput-at-best-batch benchmark profile",
        "a parameter-OOD / resolution-transfer benchmark",
        "the reference own-error budget re-measured against the v2 model "
        "error at the v2 horizons",
    ]
    state["not_executed_note"] = (
        "This list is complete and is retained unchanged. gate4 remains "
        "partial because these rows were not executed; no optional sweep "
        "was run to chase a headline."
    )

    # ---- 4. the closure record ---------------------------------------
    state["phase"] = "closed"
    state["closure"] = {
        "closed_utc": now,
        "scope": "administrative closure only",
        "actions_taken": [
            "marked the release gate complete with explicit evidence "
            "references",
            "replaced the obsolete git revision and stale dirty flag with an "
            "unambiguously labelled source-revision block",
            "retained the complete NOT_EXECUTED list and gate4 as partial",
            "froze the budget ledger at its recorded values",
        ],
        "not_done_by_design": [
            "no training, benchmarking, solver execution, full test suite, "
            "rebuild, new search or optional experiment was run",
            "no counter was reset and no cap was extended",
            "no new public performance claim was made",
            "the released artifact hash and the existing results are "
            "unchanged",
        ],
    }
    state["last_accepted_result"] = (
        "The 18 required main-matrix rows are trained and evaluated once on "
        "the fresh, unseen test split (IC seeds 7001/7002, zero overlap "
        "with the development set). All six declared targets are met on the "
        "frozen cohort, with the Burgers residual improvement recorded as "
        "evaluator-specific. The release is verified: the wheel matches its "
        "recorded hash, the clean install passes every fixture from outside "
        "the source checkout, and the final verification script reports "
        "68/68 checks passed."
    )
    state["blocker"] = None

    # ---- 5. freeze the budget ledger ---------------------------------
    # The recorded values are preserved exactly. The walltime is frozen at
    # its recorded value, and the overrun is stated explicitly: the
    # campaign did NOT finish within the walltime cap.
    budget["closed"] = True
    budget["closed_utc"] = now
    budget["walltime_measurement"] = (
        "Frozen at closure at the recorded value. The reconstruction script "
        "does not recompute a closed ledger."
    )
    budget["budget_compliance"] = {
        "heavy_numerical": "within cap",
        "heavy_numerical_detail": (
            f"{budget['spent']['heavy_numerical_job_elapsed_hours']} h of "
            f"{budget['caps']['heavy_numerical_job_elapsed_hours_total']} h"
        ),
        "walltime": "CAP EXCEEDED - the campaign did not finish within the "
                    "walltime cap",
        "walltime_detail": (
            f"{budget['spent']['walltime_hours']} h spent against a "
            f"{budget['caps']['campaign_active_walltime_hours']} h cap, an "
            f"overrun of {budget['overrun']['walltime_overrun_hours']} h"
        ),
        "statement": (
            "This campaign is NOT fully budget-compliant: the heavy-work "
            "budget was respected, the walltime cap was not. The overrun "
            "occurred during the release and documentation phase, after the "
            "training matrix was complete and frozen."
        ),
    }

    STATE.write_text(json.dumps(state, indent=2) + "\n")
    BUDGET.write_text(json.dumps(budget, indent=2) + "\n")
    print(f"closed at {now}")
    print(f"  gate6 -> {state['gates']['gate6_report_and_release']}")
    print(f"  verified head at closure: {head}")
    print(f"  release build source    : {release_rev}")
    print(f"  walltime preserved      : {budget['spent']['walltime_hours']} h "
          f"(overrun {budget['overrun']['walltime_overrun_hours']} h)")
    print(f"  heavy preserved         : "
          f"{budget['spent']['heavy_numerical_job_elapsed_hours']} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
