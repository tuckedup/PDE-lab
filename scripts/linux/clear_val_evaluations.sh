#!/usr/bin/env bash
# Remove cached validation-split evaluations so that a re-selection
# never reuses a stale artifact.
set -uo pipefail
cd "$(dirname "$0")/../.."
n=0
for f in runs/*/evaluation_val.json; do
  [ -e "$f" ] || continue
  rm -f "$f"
  n=$((n + 1))
done
echo "removed $n cached validation evaluations"
