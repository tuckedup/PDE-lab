"""Final end-to-end verification of the PDELab deliverable."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
PY = str(ROOT / ".venv/bin/python")

checks: list[tuple[str, bool, str]] = []


def run(args, label):
    proc = subprocess.run(args, capture_output=True, text=True, cwd=str(ROOT))
    return proc


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


# 1. layout
for d in ["src/pdelab", "configs/hardware", "configs/pdes", "configs/models",
          "configs/experiments", "scripts/windows", "scripts/linux",
          "tests/unit", "tests/numerical", "tests/integration",
          "data", "runs", "results", "docs"]:
    check(f"layout: {d}", (ROOT / d).exists())

for f in ["AGENTS.md", "PLAN.md", "README.md", "CITATIONS.md", "pyproject.toml",
          "configs/targets.yaml", "configs/prereg.yaml"]:
    check(f"file: {f}", (ROOT / f).exists())

for f in ["ENVIRONMENT.md", "PDE_SPEC.md", "SOLVERS.md", "DATA.md", "DECISIONS.md",
          "RESULTS.md", "REPRODUCE.md", "STATUS.md", "NEXT_SESSION.md"]:
    check(f"doc: docs/{f}", (ROOT / "docs" / f).exists())

# 2. package modules
for m in ["solvers", "data", "models", "physics", "training", "evaluation",
          "benchmarks", "resources", "reporting"]:
    check(f"package: pdelab.{m}", (ROOT / "src/pdelab" / m).is_dir())
check("package: pdelab.cli", (ROOT / "src/pdelab/cli.py").exists())

# 3. CLI commands
for cmd in [["doctor"], ["resources", "check"],
            ["solvers", "validate", "--system", "burgers"],
            ["solvers", "validate", "--system", "allen_cahn"],
            ["sweep", "--config", "configs/experiments/main_matrix.yaml",
             "--profile", "full", "--dry-run"],
            ["report"]]:
    proc = run([PY, "-m", "pdelab.cli", *cmd], " ".join(cmd))
    check(f"cli: pdelab {' '.join(cmd)}", proc.returncode == 0,
          proc.stderr.strip()[-200:])

# 4. tests and lint
proc = run([PY, "-m", "pytest", "tests/", "-q"], "pytest")
check("tests pass", proc.returncode == 0, proc.stdout.strip()[-200:])

proc = run([str(ROOT / ".venv/bin/ruff"), "check", "src/", "tests/", "scripts/"], "ruff")
check("lint clean", proc.returncode == 0, proc.stdout.strip()[-200:])

# 5. results
res = json.loads((ROOT / "results/results.json").read_text())
check("results: runs recorded", res["n_runs_completed"] >= 6, f"{res['n_runs_completed']} runs")
check("results: targets table present", len(res["targets"]) == 6, f"{len(res['targets'])} targets")
for t in res["targets"]:
    check(f"target {t['target']}: {t['status']}", t["status"] in
          ("met", "missed", "not_run", "not_comparable"),
          f"{t['measured']} (threshold {t['threshold']})")

# 6. plots
plots = list((ROOT / "results/plots").glob("*.png")) if (ROOT / "results/plots").exists() else []
check("plots generated", len(plots) >= 5, f"{len(plots)} plots")

# 7. run artifacts
runs = [d for d in (ROOT / "runs").iterdir() if d.is_dir()]
ok_runs = 0
for d in runs:
    if (d / "manifest.json").exists() and (d / "evaluation.json").exists():
        ok_runs += 1
check("runs have manifest + evaluation", ok_runs >= 6, f"{ok_runs}/{len(runs)}")

# 8. git
proc = run(["git", "status", "--porcelain"], "git status")
check("git tree clean", proc.stdout.strip() == "", proc.stdout[:200])
proc = run(["git", "log", "--oneline"], "git log")
check("git has commits", "PDELab" in proc.stdout)

# 9. no remote
proc = run(["git", "remote", "-v"], "git remote")
check("git has no remote (nothing pushed)", proc.stdout.strip() == "",
      proc.stdout[:200])

# summary
print(f"{'check':62s} {'result':>8s}  detail")
print("-" * 100)
n_fail = 0
for name, ok, detail in checks:
    if not ok:
        n_fail += 1
    print(f"{name:62s} {'PASS' if ok else 'FAIL':>8s}  {detail[:28]}")
print("-" * 100)
print(f"{len(checks) - n_fail}/{len(checks)} checks passed")
sys.exit(0 if n_fail == 0 else 1)
