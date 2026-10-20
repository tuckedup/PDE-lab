#!/usr/bin/env bash
# Run a sequence of PDELab jobs sequentially.
#
# The project job lock enforces that only one PDELab job runs at a time.
# This script additionally keeps the sequence explicit and stops on the
# first failure so that a partial run is never mistaken for a complete
# one.
#
# Usage: run_sequence.sh <command> [args...]
#   e.g. run_sequence.sh pdelab train --config configs/experiments/x.yaml --profile prototype

set -euo pipefail

cd "$(dirname "$0")/../.."

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "error: $PY not found. Create the environment first:" >&2
  echo "  uv venv --python 3.11 && uv pip install -e ." >&2
  exit 1
fi

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) :: $* ==="
"$PY" -m pdelab.cli "$@"
echo "--- exit=$? ---"
