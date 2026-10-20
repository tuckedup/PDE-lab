#!/usr/bin/env bash
# Verify that regenerating the report twice over unchanged artifacts
# produces an identical file (so the generated table is stable in
# version control).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"

"$PY" -m pdelab.cli report > /dev/null 2>&1
cp results/RESULTS.md /tmp/results_first.md
cp results/results.json /tmp/results_first.json
"$PY" -m pdelab.cli report > /dev/null 2>&1

if diff -q /tmp/results_first.md results/RESULTS.md > /dev/null && \
   diff -q /tmp/results_first.json results/results.json > /dev/null; then
  echo "OK: the generated report is deterministic over unchanged artifacts"
  exit 0
fi
echo "FAIL: the generated report changed between two runs"
diff /tmp/results_first.md results/RESULTS.md | head -20
exit 1
