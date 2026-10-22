"""Final end-to-end verification of the v2 deliverable.

Extends the layout/CLI checks with the v2-specific invariants:
  * the fresh test set is disjoint from the development set,
  * the observation times are exact,
  * the 18 required matrix rows exist and are evaluated,
  * every declared target has a valid, evidence-bound verdict,
  * the release artifact exists and matches its recorded hash,
  * the campaign state is reconstructed (not a stale placeholder).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
PY = str(ROOT / ".venv/bin/python")

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def read(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


# ---- layout and documents ----
for d in ["src/pdelab", "configs/pdes", "configs/experiments", "tests/unit",
          "tests/numerical", "tests/integration", "data", "runs", "results",
          "docs", "campaigns/v2-recovery", "campaigns/evidence-cohort",
          "campaigns/release", "campaigns/audit-verification"]:
    check(f"layout: {d}", (ROOT / d).exists())
for f in ["AGENTS.md", "PLAN.md", "README.md", "CITATIONS.md", "pyproject.toml",
          "RELEASE_MANIFEST.json", "configs/targets.yaml", "configs/prereg.yaml",
          "configs/protocol_v2.yaml", "configs/lambda_selection.json",
          "docs/ENVIRONMENT.md", "docs/PDE_SPEC.md", "docs/SOLVERS.md",
          "docs/DATA.md", "docs/DECISIONS.md", "docs/RESULTS.md",
          "docs/REPRODUCE.md", "docs/STATUS.md", "docs/NEXT_SESSION.md",
          "docs/LEGACY_INVALIDATION.md", "docs/RELEASE.md"]:
    check(f"file: {f}", (ROOT / f).exists())

# ---- CLI ----
for cmd in [["doctor"], ["resources", "check"],
            ["sweep", "--config", "configs/experiments/main_matrix.yaml",
             "--profile", "full", "--dry-run"], ["report"]]:
    p = subprocess.run([PY, "-m", "pdelab.cli", *cmd], capture_output=True,
                       text=True, cwd=str(ROOT))
    check(f"cli: pdelab {' '.join(cmd)}", p.returncode == 0, p.stderr[-160:])

for system in ("burgers", "allen_cahn"):
    p = subprocess.run(
        [PY, "-m", "pdelab.cli", "solvers", "validate", "--system", system],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    check(f"cli: pdelab solvers validate --system {system}", p.returncode == 0)

# ---- the fresh, disjoint test set ----
for system in ("burgers", "allen_cahn"):
    man = read(ROOT / f"data/{system}/{system}_manifest.json")
    if man is None:
        check(f"data: {system} manifest", False)
        continue
    recs = man["trajectories"]
    dev = {r["ic_seed"] for r in recs if r["split"] in ("train", "val")}
    test = {r["ic_seed"] for r in recs if r["split"] == "test"}
    check(f"data: {system} test IC seeds disjoint from dev", not (dev & test),
          f"overlap={len(dev & test)}")
    times = man["system"]["observation_times"]
    worst = 0.0
    for r in recs:
        a = r.get("actual_observation_times") or []
        if len(a) != len(times):
            worst = float("inf")
            break
        worst = max(worst, max(abs(x - t) for x, t in zip(a, times)))
    check(f"data: {system} observation times exact", worst <= 1e-12, f"max err {worst:.2e}")
    check(f"data: {system} test split has its own seed",
          man.get("test_split_seed") is not None
          and man["test_split_seed"] != man["split_seed"],
          f"dev={man.get('split_seed')} test={man.get('test_split_seed')}")

# ---- the 18 required matrix rows ----
# The required matrix is 2 PDEs x 3 conditions x 3 seeds, where the
# three conditions are: data-only FNO, physics FNO and data-only U-Net.
# The physics-regularized U-Net is the *optional* extension and is
# NOT part of the 18.
REQUIRED = []
for system, fno, unet in (("burgers", "fno1d", "unet1d"),
                           ("allen_cahn", "fno2d", "unet2d")):
    for arch, cond in ((fno, "data_only"), (fno, "physics"), (unet, "data_only")):
        for seed in (0, 1, 2):
            REQUIRED.append(f"{system}_{arch}_{cond}_seed{seed}")
missing = [r for r in REQUIRED if not (ROOT / "runs" / r / "evaluation.json").exists()]
check(f"matrix: all {len(REQUIRED)} required rows evaluated", not missing,
      f"missing={missing}")

# ---- the report's verdicts ----
res = read(ROOT / "results/results.json")
if res is None:
    check("report: results.json present", False)
else:
    targets = res.get("targets", [])
    check("report: six declared targets", len(targets) == 6, f"{len(targets)}")
    for t in targets:
        check(f"target {t['target']}: {t['status']}",
              t["status"] in ("met", "missed", "incomplete_cohort", "not_run",
                              "not_comparable", "stale", "invalid_protocol",
                              "failed", "interrupted"),
              f"measured={t.get('measured')}")
    valid = sum(1 for r in res.get("runs", [])
                if r.get("evidence") == "valid")
    check("report: 18 valid evidence rows", valid >= 18, f"{valid}")

# ---- the release ----
man = read(ROOT / "RELEASE_MANIFEST.json")
if man is None:
    check("release: manifest present", False)
else:
    check("release: clean install status pass",
          (man.get("clean_install") or {}).get("status") == "pass",
          str((man.get("clean_install") or {}).get("status")))
    check("release: no open blockers",
          not (man.get("clean_install") or {}).get("open_blockers"),
          str((man.get("clean_install") or {}).get("open_blockers")))
    artifact = man.get("artifact", {})
    wheel = ROOT / artifact.get("wheel_path", "")
    check("release: wheel on disk", wheel.exists(), str(wheel))
    if wheel.exists():
        import hashlib
        h = hashlib.sha256(wheel.read_bytes()).hexdigest()
        check("release: wheel hash matches the manifest",
              h == artifact.get("wheel_sha256"), h[:16])

# ---- campaign state ----
state = read(ROOT / "campaigns/v2-recovery/STATE.json")
budget = read(ROOT / "campaigns/v2-recovery/BUDGET.json")
if state is None or budget is None:
    check("campaign: state and budget present", False)
else:
    check("campaign: state is not a stale placeholder",
          state.get("phase") != "gate2"
          and any(k.startswith("gate2") for k in state.get("gates", {})),
          f"phase={state.get('phase')}")
    spent = budget.get("spent", {})
    check("campaign: budget counters are populated",
          spent.get("heavy_numerical_job_elapsed_hours", 0) > 0
          and spent.get("training_attempts", 0) > 0,
          f"heavy={spent.get('heavy_numerical_job_elapsed_hours')}h "
          f"attempts={spent.get('training_attempts')}")
    check("campaign: heavy budget respected",
          spent.get("heavy_numerical_job_elapsed_hours", 99) <= 6.0,
          f"{spent.get('heavy_numerical_job_elapsed_hours')} h of 6 h")
    matrix = read(ROOT / "campaigns/v2-recovery/MATRIX.json") or {}
    check("campaign: matrix rows recorded", len(matrix.get("rows", [])) >= 18,
          f"{len(matrix.get('rows', []))}")
    events = (ROOT / "campaigns/v2-recovery/EVENTS.jsonl").read_text().strip()
    check("campaign: events recorded", len(events.splitlines()) > 0,
          f"{len(events.splitlines())} events")

# ---- git ----
p = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                   text=True, cwd=str(ROOT))
check("git: working tree clean", p.stdout.strip() == "", p.stdout[:200])
p = subprocess.run(["git", "log", "--oneline"], capture_output=True,
                   text=True, cwd=str(ROOT))
check("git: commits exist", "PDELab" in p.stdout)
p = subprocess.run(["git", "remote", "-v"], capture_output=True,
                   text=True, cwd=str(ROOT))
check("git: no remote (nothing pushed)", p.stdout.strip() == "")

print(f"{'check':58s} {'result':>7s}  detail")
print("-" * 100)
fails = 0
for name, ok, detail in checks:
    if not ok:
        fails += 1
    print(f"{name:58s} {'PASS' if ok else 'FAIL':>7s}  {detail[:36]}")
print("-" * 100)
print(f"{len(checks)-fails}/{len(checks)} checks passed")
sys.exit(0 if fails == 0 else 1)
