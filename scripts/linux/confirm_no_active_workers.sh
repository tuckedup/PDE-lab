#!/usr/bin/env bash
# Confirm that no PDELab-owned worker is still running.
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "=== PDELab-owned processes ==="
found=0
for pat in "pdelab" "run_batch" "release_all" "evaluate_frozen" "validate_solvers"; do
  out=$(ps -eo pid,etimes,cmd | grep -F "$pat" | grep -v grep || true)
  if [ -n "$out" ]; then
    echo "$out"
    found=1
  fi
done
[ "$found" -eq 0 ] && echo "(none)"

echo
echo "=== project job lock ==="
python3 - <<'PY'
import fcntl
f = open("runs/.pdelab.lock", "a+")
try:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("lock is FREE - no PDELab job is running")
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
except OSError:
    print("lock is HELD - a job is running")
PY

echo
echo "=== GPU state ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "compute apps:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

echo
echo "=== host memory (WSL view) ==="
free -g | head -2
