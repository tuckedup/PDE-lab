#!/usr/bin/env bash
# Check the numerical-validation CLI dispatch for both systems.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"

for system in burgers allen_cahn; do
  out="/tmp/sv_${system}.json"
  if "$PY" -m pdelab.cli solvers validate --system "$system" > "$out" 2>&1; then
    echo "PASS  pdelab solvers validate --system $system (exit 0, $(wc -c < "$out") bytes)"
  else
    rc=$?
    echo "FAIL  pdelab solvers validate --system $system (exit $rc)"
    tail -5 "$out"
  fi
done
