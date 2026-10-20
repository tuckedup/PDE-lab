#!/usr/bin/env bash
# Commit the current state (local repository only; nothing is pushed).
set -uo pipefail
cd "$(dirname "$0")/../.."

git add -A
if git diff --cached --quiet; then
  echo "nothing to commit; working tree clean"
else
  git -c commit.gpgsign=false commit -q -m "${1:-PDELab: update artifacts and generated report}"
  echo "committed"
fi
echo "--- log ---"
git log --oneline -5
echo "--- uncommitted files: $(git status --porcelain | wc -l) ---"
echo "--- remotes (must be empty: nothing is pushed) ---"
git remote -v
