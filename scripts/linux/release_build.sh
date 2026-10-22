#!/usr/bin/env bash
# Build the PDELab release wheel into the project-local `dist/` directory.
#
# The wheel is built with the *project's own* `.venv` and with the build
# isolation disabled, so that
#
#   * the artifact is produced by exactly the toolchain the study used,
#   * no network access is needed (setuptools is already in `.venv`),
#   * only `build` itself is added to `.venv` when it is missing.
#
# After the build the *contents* of the wheel are inspected explicitly
# and the script FAILS if the archive contains anything that must not be
# shipped: datasets, checkpoints, run artifacts, plots, secrets, the
# virtual environment or the `campaigns/` scratch tree.
#
# Usage:
#   scripts/linux/release_build.sh
#
# Outputs (all under the repository):
#   dist/pdelab-<version>-py3-none-any.whl
#   dist/pdelab-<version>-py3-none-any.whl.sha256
#   campaigns/release/build.log
#   campaigns/release/wheel_contents.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
DIST="$ROOT/dist"
EVIDENCE="$ROOT/campaigns/release"

export PATH="$HOME/.local/bin:$PATH"

log() { printf '[release_build] %s\n' "$*"; }

if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found. The release wheel is built with the project" >&2
  echo "       virtual environment on purpose; it is missing." >&2
  exit 1
fi

mkdir -p "$DIST" "$EVIDENCE"

# ----------------------------------------------------------------------
# 1. the build frontend
# ----------------------------------------------------------------------
if ! "$PY" -c "import build" >/dev/null 2>&1; then
  log "installing 'build' into the project .venv (the only authorised dependency change)"
  if ! uv pip install --offline build --python "$PY"; then
    log "offline install failed; retrying with the index"
    uv pip install build --python "$PY"
  fi
fi
"$PY" - <<'PY'
import build
print(f"[release_build] build {build.__version__}")
PY

# ----------------------------------------------------------------------
# 2. build
# ----------------------------------------------------------------------
VERSION="$("$PY" -c "import tomllib,pathlib;print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")"
log "building pdelab $VERSION"

# Make the archive byte-reproducible: the zip member timestamps are taken
# from SOURCE_DATE_EPOCH, which is pinned to the commit date of HEAD.
# Two builds of the same tree then produce the same SHA-256, so the
# checksum recorded in RELEASE_MANIFEST.json stays valid.
if [ -z "${SOURCE_DATE_EPOCH:-}" ]; then
  SOURCE_DATE_EPOCH="$(git -C "$ROOT" log -1 --pretty=%ct 2>/dev/null || echo 0)"
  export SOURCE_DATE_EPOCH
fi
log "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH"

# Remove a wheel of the same name so that the checksum and the manifest
# can never refer to a stale artifact.
rm -f "$DIST"/pdelab-"$VERSION"-*.whl "$DIST"/pdelab-"$VERSION"-*.whl.sha256

# Remove the stale setuptools build tree.  `build/lib` is a *copy* of
# `src/pdelab`; if a module was deleted from `src/` the copy would
# survive and would be packaged into the wheel.  The directory is a
# regenerable build artifact (it is in .gitignore), so removing it makes
# the build hermetic and the archive an exact image of `src/`.
rm -rf "$ROOT/build"

"$PY" -m build --wheel --no-isolation --outdir "$DIST" . 2>&1 | tee "$EVIDENCE/build.log"

WHEEL="$(ls -1 "$DIST"/pdelab-"$VERSION"-*.whl | head -1)"
if [ ! -f "$WHEEL" ]; then
  echo "ERROR: no wheel was produced in $DIST" >&2
  exit 1
fi

# ----------------------------------------------------------------------
# 3. inspect the contents explicitly
# ----------------------------------------------------------------------
log "listing the wheel contents"
unzip -l "$WHEEL" | tee "$EVIDENCE/wheel_contents.txt"

SIZE_BYTES="$(stat -c %s "$WHEEL")"
SHA256="$(sha256sum "$WHEEL" | cut -d' ' -f1)"
printf '%s  %s\n' "$SHA256" "$(basename "$WHEEL")" > "$WHEEL.sha256"

log "wheel        : $WHEEL"
log "size (bytes) : $SIZE_BYTES"
log "sha256       : $SHA256"

# ----------------------------------------------------------------------
# 4. refuse a wheel that ships something it must not ship
# ----------------------------------------------------------------------
# The patterns are matched against every path inside the archive.  The
# top-level directories are anchored: `pdelab/data/` is the library's own
# sub-package, while a top-level `data/` would be the dataset tree.
FORBIDDEN='^(data|runs|results|configs|campaigns|dist|build|\.venv|venv)/|\.h5$|\.hdf5$|\.npz$|\.npy$|\.pt$|\.pth$|\.ckpt$|\.safetensors$|\.onnx$|\.png$|\.pdf$|\.gif$|\.svg$|(^|/)\.env$|\.key$|(^|/)credentials|(^|/)\.pdelab\.lock$'

LISTING="$EVIDENCE/wheel_contents.txt"
VIOLATIONS="$(awk 'NR>3 {print $4}' "$LISTING" | grep -E "$FORBIDDEN" || true)"

if [ -n "$VIOLATIONS" ]; then
  echo "ERROR: the wheel contains entries that must not be shipped:" >&2
  printf '%s\n' "$VIOLATIONS" >&2
  exit 1
fi

# The wheel must contain the library and the console-script metadata, and
# nothing that is not Python source or wheel metadata.
NON_CODE="$(awk 'NR>3 {print $4}' "$LISTING" | grep -Ev '\.py$|\.dist-info/|^$' || true)"
if [ -n "$NON_CODE" ]; then
  echo "ERROR: the wheel contains non-source entries:" >&2
  printf '%s\n' "$NON_CODE" >&2
  exit 1
fi

N_ENTRIES="$(awk 'NR>3 && NF>=4 {n++} END {print n+0}' "$LISTING")"
log "content check passed: $N_ENTRIES entries, no datasets, checkpoints,"
log "secrets, plots, .venv or campaigns/ inside the archive"

# ----------------------------------------------------------------------
# 5. a machine readable summary
# ----------------------------------------------------------------------
"$PY" - "$WHEEL" "$SIZE_BYTES" "$SHA256" "$N_ENTRIES" "$EVIDENCE/build_summary.json" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

wheel, size, sha, n_entries, out = sys.argv[1:6]
with zipfile.ZipFile(wheel) as zf:
    names = sorted(zf.namelist())
    uncompressed = sum(i.file_size for i in zf.infolist())
payload = {
    "wheel_path": str(Path(wheel).relative_to(Path.cwd())),
    "wheel_filename": Path(wheel).name,
    "wheel_size_bytes": int(size),
    "wheel_uncompressed_bytes": int(uncompressed),
    "wheel_sha256": sha,
    "wheel_entry_count": int(n_entries),
    "wheel_entries": names,
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"[release_build] wrote {out}")
PY

log "done"
