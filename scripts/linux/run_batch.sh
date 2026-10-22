#!/usr/bin/env bash
# Run a list of experiment configs sequentially, with a hard per-job
# wall-time limit and a hard stop on the first failure.
#
# Usage: run_batch.sh <config-name> [config-name ...]
#
# Every job goes through the CLI, which takes the project lease and runs
# the resource preflight. The batch stops on the first failure so a
# partial batch is never mistaken for a complete one.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"

MAX_JOB_MIN="${MAX_JOB_MIN:-45}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <config-name> [config-name ...]" >&2
  exit 2
fi

overall=0
mkdir -p runs/_logs
for name in "$@"; do
  cfg="configs/experiments/${name}.yaml"
  if [ ! -f "$cfg" ]; then
    echo ">>> SKIP $name: no config at $cfg" >&2
    overall=1
    continue
  fi
  log="runs/_logs/${name}.log"
  echo ">>> $(date -u +%Y-%m-%dT%H:%M:%SZ) training $name (max ${MAX_JOB_MIN} min, log $log)"
  start=$(date +%s)
  if timeout "${MAX_JOB_MIN}m" "$PY" -m pdelab.cli train --config "$cfg" --profile prototype > "$log" 2>&1; then
    mins=$(( ($(date +%s)-start)/60 ))
    echo "<<< $(date -u +%Y-%m-%dT%H:%M:%SZ) done $name (${mins} min): $(grep -E 'best_val_loss' "$log" | tail -1 | sed 's/^ *//')"
  else
    rc=$?
    mins=$(( ($(date +%s)-start)/60 ))
    echo "<<< FAILED $name (exit $rc, ${mins} min) - see $log" >&2
    tail -5 "$log" >&2
    overall=1
    break
  fi
done
exit $overall
