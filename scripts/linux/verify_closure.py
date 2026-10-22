"""Metadata-only verification of the closed campaign.

This script deliberately runs **no** solver, no benchmark, no report
regeneration and no test suite. It only inspects files, hashes and the
recorded campaign state.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
CAMPAIGN = ROOT / "campaigns" / "v2-recovery"

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def read(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


state = read(CAMPAIGN / "STATE.json")
budget = read(CAMPAIGN / "BUDGET.json")
manifest = read(ROOT / "RELEASE_MANIFEST.json")

# ---- the closure itself ----
check("closure: STATE.json present", state is not None)
if state:
    check("closure: phase is closed", state.get("phase") == "closed",
          str(state.get("phase")))
    check("closure: release gate complete",
          state.get("gates", {}).get("gate6_report_and_release") == "complete",
          str(state.get("gates", {}).get("gate6_report_and_release")))
    check("closure: gate6 carries explicit evidence references",
          bool(state.get("gates", {}).get("gate6_evidence")))
    check("closure: gate4 left partial",
          str(state.get("gates", {}).get("gate4_ablations", "")).startswith("partial"),
          str(state.get("gates", {}).get("gate4_ablations")))
    check("closure: NOT_EXECUTED list retained and non-empty",
          len(state.get("not_executed", [])) > 0,
          f"{len(state.get('not_executed', []))} items")
    check("closure: scientific caveats retained",
          len(state.get("known_limitations", [])) >= 4,
          f"{len(state.get('known_limitations', []))} limitations")
    g = state.get("git", {})
    check("closure: no stale dirty flag on the verified head",
          g.get("verified_head_working_tree_clean") is True,
          str(g.get("verified_head_working_tree_clean")))
    check("closure: source revision labelled unambiguously",
          "verified_head_at_closure" in g and "release_build_source_revision" in g
          and "labelling_note" in g,
          f"head={str(g.get('verified_head_at_closure'))[:12]}")
    check("closure: no self-referential 'current revision' claim",
          "revision" not in g or g.get("revision") is None,
          "git.revision removed in favour of labelled fields")

# ---- the frozen ledger ----
check("ledger: BUDGET.json present", budget is not None)
if budget:
    s = budget.get("spent", {})
    o = budget.get("overrun", {})
    check("ledger: recorded walltime preserved at 8.355 h",
          s.get("walltime_hours") == 8.355, str(s.get("walltime_hours")))
    check("ledger: 0.355 h overrun preserved",
          o.get("walltime_overrun_hours") == 0.355, str(o.get("walltime_overrun_hours")))
    check("ledger: heavy charges preserved",
          s.get("heavy_numerical_job_elapsed_hours") == 5.3271
          and s.get("heavy_breakdown_hours", {}).get("training") == 5.1485,
          f"{s.get('heavy_numerical_job_elapsed_hours')} h")
    check("ledger: training attempts not reset",
          s.get("training_attempts") == 20, str(s.get("training_attempts")))
    check("ledger: caps not extended",
          budget.get("caps", {}).get("campaign_active_walltime_hours") == 8.0
          and budget.get("caps", {}).get("heavy_numerical_job_elapsed_hours_total") == 6.0)
    check("ledger: closed flag set", budget.get("closed") is True)
    check("ledger: budget compliance is not claimed",
          "CAP EXCEEDED" in budget.get("budget_compliance", {}).get("walltime", ""),
          budget.get("budget_compliance", {}).get("walltime", ""))

# ---- the released artifact is preserved ----
if manifest:
    art = manifest.get("artifact", {})
    wheel = ROOT / art.get("wheel_path", "")
    check("release: wheel still present", wheel.exists(), str(wheel))
    if wheel.exists():
        h = hashlib.sha256(wheel.read_bytes()).hexdigest()
        check("release: wheel hash preserved",
              h == art.get("wheel_sha256"), h[:16])
    check("release: clean install still recorded as pass",
          (manifest.get("clean_install") or {}).get("status") == "pass")
    check("release: no open blockers",
          not (manifest.get("clean_install") or {}).get("open_blockers"))

# ---- results and evidence paths ----
for p in ["docs/RESULTS.md", "results/results.json", "docs/STATUS.md",
          "docs/LEGACY_INVALIDATION.md", "docs/RELEASE.md",
          "RELEASE_MANIFEST.json", "campaigns/v2-recovery/MATRIX.json",
          "campaigns/v2-recovery/EVENTS.jsonl",
          "campaigns/evidence-cohort/cohort.yaml",
          "campaigns/audit-verification/audit_reproduction.json",
          "campaigns/release/clean_install.json",
          "campaigns/release/build_reproducibility.json"]:
    check(f"evidence: {p}", (ROOT / p).exists())

res = read(ROOT / "results/results.json")
if res:
    targets = res.get("targets", [])
    check("results: six target verdicts retained", len(targets) == 6, f"{len(targets)}")
    check("results: no target lost its status",
          all(t.get("status") for t in targets),
          ", ".join(f"{t['target']}={t['status']}" for t in targets))

# ---- git ----
p = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                   text=True, cwd=str(ROOT))
check("git: working tree clean", p.stdout.strip() == "", p.stdout[:160])
p = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True,
                   cwd=str(ROOT))
check("git: no remote (nothing pushed)", p.stdout.strip() == "")

print(f"{'check':58s} {'result':>7s}  detail")
print("-" * 100)
fails = sum(1 for _n, ok, _d in checks if not ok)
for name, ok, detail in checks:
    print(f"{name:58s} {'PASS' if ok else 'FAIL':>7s}  {detail[:36]}")
print("-" * 100)
print(f"{len(checks)-fails}/{len(checks)} checks passed")
sys.exit(0 if fails == 0 else 1)
