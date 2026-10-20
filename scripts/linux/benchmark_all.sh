#!/usr/bin/env bash
# Run the matched inference benchmark on every completed run that has no
# benchmark artifact yet.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"

for d in runs/*/; do
  run_id=$(basename "$d")
  [ -f "$d/manifest.json" ] || continue
  [ -f "$d/benchmark_matched.json" ] && continue
  echo ">>> benchmarking $run_id"
  "$PY" -m pdelab.cli benchmark --run-id "$run_id" --profile matched > /dev/null 2>&1 \
    || echo "    FAILED"
done

echo
"$PY" scripts/linux/summarize_bench.py burgers_fno1d_data_only_seed0
