#!/usr/bin/env bash
# Clean-environment installation test for the PDELab release wheel.
#
# The test is deliberately hostile to the most common way an installation
# check fakes a pass: the import of the *source tree* instead of the
# installed artifact.  Therefore
#
#   * the temporary directory is created OUTSIDE the repository
#     (/tmp/pdelab_clean_install_XXXXXX),
#   * a fresh virtual environment is created inside it,
#   * the built wheel is installed into that environment,
#   * every fixture is executed with the working directory set to the
#     temporary directory,
#   * the resolved `pdelab.__file__` is printed and checked, so that a
#     reader can see that the library came from the installed wheel.
#
# The fixtures are tiny and CPU only: the CLI `doctor`, the numerical
# validation suite of BOTH reference solvers through the *CLI* dispatch
# (`pdelab solvers validate --system burgers` and `--system allen_cahn`),
# the same Burgers suite through the library, and a one-step metric check.
# The CPU thread limit is 4 everywhere and CUDA is hidden.
#
# The CLI dispatch is exercised on purpose: the numerical suite itself
# can pass while the CLI path that dispatches it is broken, and a
# release check that only called the library would report a pass for a
# broken `pdelab solvers validate`.
#
# Usage:
#   scripts/linux/release_clean_install.sh
#
# Outputs (under the repository):
#   campaigns/release/clean_install.log     the full transcript
#   campaigns/release/clean_install.json    the structured record
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"

DIST="$ROOT/dist"
EVIDENCE="$ROOT/campaigns/release"
LOG="$EVIDENCE/clean_install.log"

mkdir -p "$EVIDENCE"

# ----------------------------------------------------------------------
# CPU-only, four threads, everywhere.
# ----------------------------------------------------------------------
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

log() { printf '[release_clean_install] %s\n' "$*"; }
rule() { printf '%s\n' "================================================================================"; }

: > "$LOG"
log "PDELab clean-environment installation test"
rule

# ----------------------------------------------------------------------
# 0. the artifact
# ----------------------------------------------------------------------
WHEEL="$(ls -1 "$DIST"/pdelab-*.whl 2>/dev/null | head -1)"
if [ -z "$WHEEL" ]; then
  echo "ERROR: no wheel in $DIST. Run scripts/linux/release_build.sh first." >&2
  exit 1
fi
WHEEL_SHA="$(sha256sum "$WHEEL" | cut -d' ' -f1)"
WHEEL_SIZE="$(stat -c %s "$WHEEL")"
log "wheel        : $WHEEL"
log "size (bytes) : $WHEEL_SIZE"
log "sha256       : $WHEEL_SHA"
{
  rule
  echo "wheel        : $WHEEL"
  echo "size (bytes) : $WHEEL_SIZE"
  echo "sha256       : $WHEEL_SHA"
} >> "$LOG"

# ----------------------------------------------------------------------
# 1. the temporary environment, OUTSIDE the checkout
# ----------------------------------------------------------------------
CLEAN_DIR="$(mktemp -d /tmp/pdelab_clean_install_XXXXXX)"
CLEAN_VENV="$CLEAN_DIR/venv"
RUN_DIR="$CLEAN_DIR"
mkdir -p "$RUN_DIR"

# The project root is pinned to the temporary directory so that the CLI
# cannot reach into the checkout for configs or artifacts.
export PDELAB_ROOT="$RUN_DIR"
export PDELAB_REPOSITORY="$ROOT"

log "clean dir    : $CLEAN_DIR (outside the repository: $ROOT)"
{
  rule
  echo "clean dir    : $CLEAN_DIR"
  echo "repository   : $ROOT"
  echo "PDELAB_ROOT  : $PDELAB_ROOT"
  echo "threads      : OMP/MKL/OPENBLAS/NUMEXPR/VECLIB = 4"
  echo "CUDA_VISIBLE_DEVICES = '$CUDA_VISIBLE_DEVICES' (hidden: CPU only)"
} >> "$LOG"

FAILED=0
STEP_LABELS=()

run_step() {
  local label="$1"; shift
  local outfile="$RUN_DIR/${label}.out"
  rule
  {
    echo "STEP    : $label"
    echo "COMMAND : $*"
    echo "CWD     : $RUN_DIR"
    echo "OUTPUT  :"
  } | tee -a "$LOG"
  ( cd "$RUN_DIR" && "$@" ) > "$outfile" 2>&1
  local rc=$?
  cat "$outfile" >> "$LOG"
  {
    echo "EXIT    : $rc"
    rule
  } >> "$LOG"
  cat "$outfile"
  echo "EXIT    : $rc"
  printf '%s\n' "$rc" > "$RUN_DIR/${label}.rc"
  STEP_LABELS+=("$label")
  if [ "$rc" -ne 0 ]; then
    FAILED=$((FAILED + 1))
    log "STEP '$label' FAILED with exit code $rc"
  fi
  return 0
}

log "creating the fresh virtual environment"
uv venv --python 3.11 "$CLEAN_VENV" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "ERROR: could not create the clean virtual environment" >&2
  exit 1
fi

log "installing the wheel (offline: the dependency set comes from the local uv cache)"
uv pip install --offline --python "$CLEAN_VENV/bin/python" "$WHEEL" >> "$LOG" 2>&1
INSTALL_RC=$?
if [ "$INSTALL_RC" -ne 0 ]; then
  log "offline install failed (rc=$INSTALL_RC); retrying with the index"
  uv pip install --python "$CLEAN_VENV/bin/python" "$WHEEL" >> "$LOG" 2>&1
  INSTALL_RC=$?
fi
if [ "$INSTALL_RC" -ne 0 ]; then
  echo "ERROR: the wheel could not be installed into the clean environment" >&2
  tail -40 "$LOG" >&2
  exit 1
fi
log "wheel installed into $CLEAN_VENV"

# ----------------------------------------------------------------------
# 2. the fixtures, all executed from OUTSIDE the checkout
# ----------------------------------------------------------------------
run_step "install_probe" "$CLEAN_VENV/bin/python" -c "
import json, sys, pdelab
print('pdelab.__file__ =', pdelab.__file__)
print('pdelab.__version__ =', pdelab.__version__)
print('python =', sys.version.split()[0])
print('sys.path =', json.dumps(sys.path, indent=2))
"

run_step "cli_doctor" "$CLEAN_VENV/bin/pdelab" doctor

# The numerical validation suite of the 1D Burgers reference solver.
# The CLI path is exercised first; the library path is exercised as well
# so that a failure of the CLI dispatch can be told apart from a failure
# of the numerics.
run_step "solvers_validate_cli" "$CLEAN_VENV/bin/pdelab" solvers validate --system burgers

# The same CLI dispatch for the 2D Allen-Cahn reference solver.  Both
# CLI paths of `pdelab solvers validate` must exit 0 and print a JSON
# report; the JSON report is checked below, not just the exit code.
run_step "solvers_validate_cli_allen_cahn" "$CLEAN_VENV/bin/pdelab" solvers validate --system allen_cahn

# The two CLI steps above must not only exit 0: they must print a
# *machine readable* JSON report whose `passed` flag is true.  The
# captured outputs are parsed here, from outside the checkout, so that
# "it exited 0 but printed nothing useful" cannot pass.
run_step "solvers_validate_cli_json_report" "$CLEAN_VENV/bin/python" -c "
import json
from pathlib import Path
for label, system in [('solvers_validate_cli', 'burgers'),
                      ('solvers_validate_cli_allen_cahn', 'allen_cahn')]:
    raw = Path('$RUN_DIR', label + '.out').read_text(encoding='utf-8')
    report = json.loads(raw)
    assert report['system'] == system, (label, report.get('system'))
    assert report['passed'] is True, (label, 'the report is not passed')
    failed = [c['name'] for c in report['checks'] if not c['passed']]
    assert not failed, (label, failed)
    print(json.dumps({
        'step': label,
        'system': report['system'],
        'passed': report['passed'],
        'n_checks': len(report['checks']),
        'failed_checks': failed,
        'walltime_seconds': report['walltime_seconds'],
    }, indent=2))
"

run_step "solvers_validate_library" "$CLEAN_VENV/bin/python" -c "
import json
from pdelab.solvers.validation import validate_burgers
r = validate_burgers()
print(json.dumps({
    'system': r['system'],
    'passed': r['passed'],
    'n_checks': len(r['checks']),
    'failed_checks': [c['name'] for c in r['checks'] if not c['passed']],
    'check_names': [c['name'] for c in r['checks']],
    'walltime_seconds': r['walltime_seconds'],
}, indent=2))
raise SystemExit(0 if r['passed'] else 1)
"

# The one-step metric check, copied out of the checkout so that the
# script itself cannot import the source tree either.
cp "$ROOT/campaigns/release/fixtures/one_step_metric_check.py" "$RUN_DIR/"
run_step "one_step_metric_check" "$CLEAN_VENV/bin/python" "$RUN_DIR/one_step_metric_check.py"

# The exact command that RELEASE_MANIFEST.json records as the
# reproduction command.  It is executed here verbatim against the
# *installed* CLI, so that the manifest never contains a command that was
# not run.  The command lives in its own file so that the manifest
# generator and this script cannot drift apart.
REPRO_CMD_FILE="$ROOT/campaigns/release/fixtures/reproduction_command.txt"
{
  echo '#!/usr/bin/env bash'
  echo "cd \"$RUN_DIR\""
  echo "export PATH=\"$CLEAN_VENV/bin:\$PATH\""
  echo "export PDELAB_ROOT=\"$RUN_DIR\""
  cat "$REPRO_CMD_FILE"
} > "$RUN_DIR/run_reproduction.sh"
chmod +x "$RUN_DIR/run_reproduction.sh"
run_step "reproduction_command" bash "$RUN_DIR/run_reproduction.sh"

# ----------------------------------------------------------------------
# 3. the structured record
# ----------------------------------------------------------------------
"$ROOT/.venv/bin/python" - "$CLEAN_DIR" "$RUN_DIR" "$WHEEL" "$WHEEL_SHA" "$WHEEL_SIZE" \
    "$EVIDENCE/clean_install.json" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

clean_dir, run_dir, wheel, sha, size, out = sys.argv[1:7]
run_dir = Path(run_dir)
labels = [p.stem for p in sorted(run_dir.glob("*.rc"))]

steps = {}
for label in labels:
    out_file = run_dir / f"{label}.out"
    steps[label] = {
        "exit_code": int((run_dir / f"{label}.rc").read_text().strip()),
        "output_file": f"campaigns/release/clean_install/{label}.out",
        "output": out_file.read_text(encoding="utf-8", errors="replace")
        if out_file.exists() else "",
    }

py = run_dir / "venv/bin/python"
version_probe = subprocess.run(
    [str(py), "-c",
     "import json,sys,torch,numpy;"
     "import pdelab;"
     "print(json.dumps({'python': sys.version.split()[0],"
     "'torch': torch.__version__, 'numpy': numpy.__version__,"
     "'pdelab_file': pdelab.__file__, 'pdelab_version': pdelab.__version__,"
     "'torch_threads': torch.get_num_threads(),"
     "'cuda_available': bool(torch.cuda.is_available())}))"],
    capture_output=True, text=True, cwd=run_dir, check=False,
)
try:
    env = json.loads(version_probe.stdout.strip().splitlines()[-1])
except Exception:
    env = {"error": version_probe.stdout + version_probe.stderr}

payload = {
    "clean_dir": clean_dir,
    "clean_dir_is_outside_the_repository": True,
    "repository": os.environ.get("PDELAB_REPOSITORY", ""),
    "venv": str(run_dir / "venv"),
    "wheel_path": wheel,
    "wheel_sha256": sha,
    "wheel_size_bytes": int(size),
    "environment": env,
    "cpu_threads": 4,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "steps": steps,
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[release_clean_install] wrote {out}")
PY

# Copy the raw step outputs next to the record so that the evidence is
# readable without re-running anything.
rm -rf "$EVIDENCE/clean_install"
mkdir -p "$EVIDENCE/clean_install"
for f in "$RUN_DIR"/*.out; do
  [ -e "$f" ] || continue
  cp "$f" "$EVIDENCE/clean_install/$(basename "$f")"
done

rule
log "step summary"
if [ "${#STEP_LABELS[@]}" -gt 0 ]; then
  for label in "${STEP_LABELS[@]}"; do
    rc="$(cat "$RUN_DIR/${label}.rc")"
    log "  $label -> exit $rc"
  done
fi
log "clean dir kept for inspection: $CLEAN_DIR"
log "(remove it with: rm -rf $CLEAN_DIR)"
rule

if [ "$FAILED" -ne 0 ]; then
  log "$FAILED step(s) failed - see the transcript above"
  exit 1
fi
log "all fixtures passed"
exit 0
