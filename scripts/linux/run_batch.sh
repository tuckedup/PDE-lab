#!/usr/bin/env bash
# Run a list of experiment configs sequentially.
#
# Usage: run_batch.sh <config-name> [config-name ...]
#
# Each config is trained with the `prototype` profile. The CLI takes the
# project job lock, so a batch cannot overlap with another PDELab job.
# The batch stops on the first failure so that a partial batch is never
# mistaken for a complete one.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <config-name> [config-name ...]" >&2
  exit 2
fi

overall=0
for name in "$@"; do
  cfg="configs/experiments/${name}.yaml"
  if [ ! -f "$cfg" ]; then
    echo ">>> SKIP $name: no config at $cfg" >&2
    overall=1
    continue
  fi
  echo ">>> $(date -u +%Y-%m-%dT%H:%M:%SZ) training $name"
  if "$PY" -m pdelab.cli train --config "$cfg" --profile prototype; then
    echo "<<< $(date -u +%Y-%m-%dT%H:%M:%SZ) done $name"
  else
    echo "<<< FAILED $name" >&2
    overall=1
  fi
done
exit $overall
