#!/usr/bin/env bash
# Evaluate every completed run that has no evaluation.json yet, then
# print a compact comparison table.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"

for d in runs/*/; do
  run_id=$(basename "$d")
  [ -f "$d/manifest.json" ] || continue
  if [ ! -f "$d/evaluation.json" ]; then
    echo ">>> evaluating $run_id"
    "$PY" -m pdelab.cli evaluate --run-id "$run_id" > /dev/null 2>&1 \
      || echo "    FAILED"
  fi
done

echo
"$PY" scripts/linux/compare_runs.py
