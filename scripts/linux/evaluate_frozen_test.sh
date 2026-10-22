#!/usr/bin/env bash
# Gate 5: evaluate the FRESH test split once, as a batch, for every run
# that is a declared member of the frozen cohort.
#
# The test set is not used for any design, tuning, lambda selection or
# stopping decision. This is the only time it is touched.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"

RUNS=$("$PY" - <<'PYEOF'
import re, sys
from pathlib import Path
text = Path("campaigns/evidence-cohort/cohort.yaml").read_text()
ids = []
for m in re.finditer(r"^      - (\S+)$", text, flags=re.M):
    name = m.group(1)
    if name.startswith(("burgers_", "allen_cahn_")) and name not in ids:
        ids.append(name)
print(" ".join(ids))
PYEOF
)

echo "cohort members: $RUNS"
echo
for name in $RUNS; do
  if [ ! -f "runs/$name/manifest.json" ]; then
    echo ">>> SKIP $name: no run" >&2
    continue
  fi
  if [ -f "runs/$name/evaluation.json" ]; then
    echo ">>> $name already evaluated; refreshing to bind the current identity"
    rm -f "runs/$name/evaluation.json" "runs/$name/evaluation.json.sha256"
  fi
  echo ">>> $(date -u +%H:%M:%SZ) evaluating $name on the fresh test split"
  if "$PY" -m pdelab.cli evaluate --run-id "$name" --split test > "runs/_logs/eval_$name.log" 2>&1; then
    echo "<<< done $name"
  else
    echo "<<< FAILED $name" >&2
    tail -5 "runs/_logs/eval_$name.log" >&2
  fi
done
echo
echo "=== final test evaluation complete ==="
