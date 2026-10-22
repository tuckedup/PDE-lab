#!/usr/bin/env bash
# Reproducibility check for the PDELab release wheel.
#
# The claim in RELEASE_MANIFEST.json is that the wheel is byte
# reproducible: with SOURCE_DATE_EPOCH pinned to the commit date of HEAD,
# consecutive builds of the same tree produce the same SHA-256.  This
# script *measures* that claim instead of asserting it:
#
#   * the build tree is cleaned before every build, so that no stale
#     `build/lib` copy can influence the archive,
#   * the wheel is built three times in a row into a temporary
#     directory OUTSIDE the repository,
#   * the three SHA-256 digests are compared with each other and with
#     the digest of the wheel that is actually shipped in `dist/`,
#   * the result is written to campaigns/release/build_reproducibility.json
#     and the script exits non-zero if the digests differ.
#
# Usage:
#   scripts/linux/release_reproducibility.sh
#
# Outputs (under the repository):
#   campaigns/release/build_reproducibility.json   the structured record
#   campaigns/release/build_reproducibility.log     the full transcript
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
DIST="$ROOT/dist"
EVIDENCE="$ROOT/campaigns/release"
LOG="$EVIDENCE/build_reproducibility.log"

mkdir -p "$EVIDENCE"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

log() { printf '[release_reproducibility] %s\n' "$*"; }

if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found; the wheel is built with the project .venv." >&2
  exit 1
fi

: > "$LOG"
{
  echo "PDELab wheel reproducibility check"
  echo "=================================="
} >> "$LOG"

# ----------------------------------------------------------------------
# the pinned epoch
# ----------------------------------------------------------------------
SOURCE_DATE_EPOCH="$(git -C "$ROOT" log -1 --pretty=%ct 2>/dev/null || echo 0)"
export SOURCE_DATE_EPOCH
COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
VERSION="$("$PY" -c "import tomllib,pathlib;print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")"

log "commit           : $COMMIT"
log "SOURCE_DATE_EPOCH: $SOURCE_DATE_EPOCH ($(date -u -d "@$SOURCE_DATE_EPOCH" '+%Y-%m-%dT%H:%M:%SZ'))"
{
  echo "commit           : $COMMIT"
  echo "SOURCE_DATE_EPOCH: $SOURCE_DATE_EPOCH ($(date -u -d "@$SOURCE_DATE_EPOCH" '+%Y-%m-%dT%H:%M:%SZ'))"
} >> "$LOG"

# ----------------------------------------------------------------------
# three consecutive builds
# ----------------------------------------------------------------------
WORK="$(mktemp -d /tmp/pdelab_reproducibility_XXXXXX)"
DIGESTS=()
SIZES=()

for i in 1 2 3; do
  OUT="$WORK/build$i"
  mkdir -p "$OUT"
  log "build $i/3 -> $OUT"
  {
    echo
    echo "---- build $i/3 ----"
  } >> "$LOG"
  # hermetic: no stale build/lib copy can leak into the archive
  rm -rf "$ROOT/build"
  if ! "$PY" -m build --wheel --no-isolation --outdir "$OUT" . >> "$LOG" 2>&1; then
    echo "ERROR: build $i failed" >&2
    tail -20 "$LOG" >&2
    exit 1
  fi
  W="$(ls -1 "$OUT"/pdelab-"$VERSION"-*.whl | head -1)"
  D="$(sha256sum "$W" | cut -d' ' -f1)"
  S="$(stat -c %s "$W")"
  DIGESTS+=("$D")
  SIZES+=("$S")
  log "  sha256 : $D"
  log "  bytes  : $S"
done

# the shipped artifact
SHIPPED=""
SHIPPED_SHA=""
SHIPPED_SIZE=""
for f in "$DIST"/pdelab-"$VERSION"-*.whl; do
  [ -e "$f" ] || continue
  SHIPPED="$f"
  SHIPPED_SHA="$(sha256sum "$f" | cut -d' ' -f1)"
  SHIPPED_SIZE="$(stat -c %s "$f")"
  break
done
if [ -z "$SHIPPED" ]; then
  log "WARNING: no shipped wheel in $DIST (run scripts/linux/release_build.sh first)"
fi

log "shipped wheel    : ${SHIPPED:-<none>}"
log "shipped sha256   : ${SHIPPED_SHA:-<none>}"

# ----------------------------------------------------------------------
# the verdict
# ----------------------------------------------------------------------
IDENTICAL=1
for d in "${DIGESTS[@]}"; do
  if [ "$d" != "${DIGESTS[0]}" ]; then IDENTICAL=0; fi
done
MATCHES_SHIPPED=0
if [ -n "$SHIPPED_SHA" ] && [ "$SHIPPED_SHA" = "${DIGESTS[0]}" ]; then
  MATCHES_SHIPPED=1
fi

log "three builds identical : $([ "$IDENTICAL" -eq 1 ] && echo yes || echo no)"
log "matches shipped wheel  : $([ "$MATCHES_SHIPPED" -eq 1 ] && echo yes || echo no)"

"$PY" - "$COMMIT" "$SOURCE_DATE_EPOCH" "$VERSION" "${DIGESTS[@]}" "${SIZES[@]}" \
    "$SHIPPED" "$SHIPPED_SHA" "$SHIPPED_SIZE" "$IDENTICAL" "$MATCHES_SHIPPED" \
    "$EVIDENCE/build_reproducibility.json" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

commit = sys.argv[1]
epoch = int(sys.argv[2])
version = sys.argv[3]
digests = sys.argv[4:7]
sizes = sys.argv[7:10]
shipped, shipped_sha, shipped_size = sys.argv[10:13]
identical, matches = sys.argv[13:15]
out = sys.argv[15]

payload = {
    "schema": "pdelab-build-reproducibility/1",
    "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "commit": commit,
    "source_date_epoch": epoch,
    "source_date_epoch_utc": datetime.fromtimestamp(
        epoch, UTC
    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "version": version,
    "builds": [
        {"index": i + 1, "sha256": d, "size_bytes": int(s)}
        for i, (d, s) in enumerate(zip(digests, sizes))
    ],
    "three_consecutive_builds_identical": identical == "1",
    "shipped_wheel_path": shipped,
    "shipped_wheel_sha256": shipped_sha,
    "shipped_wheel_size_bytes": int(shipped_size) if shipped_size else None,
    "shipped_wheel_matches_the_three_builds": matches == "1",
    "method": "three consecutive `python -m build --wheel --no-isolation` runs "
    "of the same tree with SOURCE_DATE_EPOCH pinned to the commit date of "
    "HEAD and the setuptools build tree removed before each run",
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[release_reproducibility] wrote {out}")
PY

if [ "$IDENTICAL" -ne 1 ]; then
  echo "ERROR: the three consecutive builds do NOT have the same SHA-256" >&2
  exit 1
fi
if [ -n "$SHIPPED_SHA" ] && [ "$MATCHES_SHIPPED" -ne 1 ]; then
  echo "ERROR: the shipped wheel does not match the reproduced builds" >&2
  exit 1
fi

log "work directory kept for inspection: $WORK"
log "all checks passed"
exit 0
