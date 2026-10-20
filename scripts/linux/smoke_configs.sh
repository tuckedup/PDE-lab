#!/usr/bin/env bash
# Quick configuration check: run every listed config for 2 epochs on the
# smoke profile so that a broken config is caught before a long batch.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"

overall=0
for name in "$@"; do
  cfg="configs/experiments/${name}.yaml"
  if [ ! -f "$cfg" ]; then
    echo ">>> SKIP $name: no config at $cfg" >&2
    overall=1
    continue
  fi
  echo ">>> smoke-testing $name"
  if "$PY" -m pdelab.cli train --config "$cfg" --profile smoke; then
    echo "<<< ok $name"
  else
    echo "<<< FAILED $name" >&2
    overall=1
  fi
done
exit $overall
