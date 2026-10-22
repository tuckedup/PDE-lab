#!/usr/bin/env bash
# Build, verify and describe the local PDELab release, in that order.
#
#   1. scripts/linux/release_build.sh        -> dist/*.whl + content check
#   2. scripts/linux/release_reproducibility.sh -> three-build SHA-256 check
#   3. scripts/linux/release_clean_install.sh -> clean-environment install test
#   4. scripts/linux/release_manifest.py      -> RELEASE_MANIFEST.json
#
# Stage 2 exits non-zero when a fixture fails.  The stage is still
# recorded (the structured record is written before the exit) and the
# manifest is generated anyway, so that a failure is *reported* instead
# of silently disappearing.  The exit code of this script is the number
# of failed stages.
#
# Usage:
#   scripts/linux/release_all.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"

FAILED=0

stage() {
  local name="$1"; shift
  echo
  echo "################################################################################"
  echo "# STAGE: $name"
  echo "################################################################################"
  "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "--- stage '$name' exited with $rc ---"
    FAILED=$((FAILED + 1))
  fi
  return 0
}

stage "build the wheel" bash "$ROOT/scripts/linux/release_build.sh"
stage "wheel reproducibility check" bash "$ROOT/scripts/linux/release_reproducibility.sh"
stage "clean-environment installation test" bash "$ROOT/scripts/linux/release_clean_install.sh"
stage "write the release manifest" "$ROOT/.venv/bin/python" "$ROOT/scripts/linux/release_manifest.py"

echo
echo "################################################################################"
if [ "$FAILED" -ne 0 ]; then
  echo "# RELEASE VERIFICATION: $FAILED stage(s) failed - see the output above."
  echo "# The manifest records the failure; nothing was weakened to hide it."
else
  echo "# RELEASE VERIFICATION: all stages passed."
fi
echo "################################################################################"
exit "$FAILED"
