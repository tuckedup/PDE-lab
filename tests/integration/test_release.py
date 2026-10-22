"""Fast checks of the local release bundle.

These tests are deliberately cheap: they never rebuild the wheel and
never install anything.  They only assert that the *recorded* release
manifest agrees with what is on disk.

If the wheel has not been built yet the checksum test is skipped with a
clear reason instead of failing, so that the suite stays usable in a
fresh checkout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pdelab.config import project_root

MANIFEST_NAME = "RELEASE_MANIFEST.json"


def _root() -> Path:
    return project_root()


def _manifest_path() -> Path:
    return _root() / MANIFEST_NAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    path = _manifest_path()
    if not path.exists():
        pytest.skip(
            f"{MANIFEST_NAME} does not exist; build the release with "
            "scripts/linux/release_build.sh and scripts/linux/release_manifest.py"
        )
    with path.open(encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{MANIFEST_NAME} is not valid JSON: {exc}")


def test_release_manifest_exists(manifest):
    assert _manifest_path().is_file(), f"{MANIFEST_NAME} must exist at the repository root"
    assert manifest, f"{MANIFEST_NAME} must not be empty"


def test_release_manifest_is_valid_json(manifest):
    # The fixture already parsed the file; re-parse it here so that a
    # failure points at this test.
    with _manifest_path().open(encoding="utf-8") as fh:
        payload = json.load(fh)
    assert isinstance(payload, dict)
    assert payload.get("schema"), "the manifest must declare its schema"
    assert payload.get("project", {}).get("version"), "the manifest must declare a version"


def test_recorded_sha256_matches_the_wheel_on_disk(manifest):
    artifact = manifest.get("artifact", {})
    wheel_name = artifact.get("wheel_filename")
    recorded = artifact.get("wheel_sha256")
    assert wheel_name, "the manifest must record the wheel filename"
    assert recorded, "the manifest must record the wheel SHA-256"

    wheel = _root() / "dist" / wheel_name
    if not wheel.exists():
        pytest.skip(
            f"the wheel {wheel} is not on disk; run scripts/linux/release_build.sh "
            "to rebuild it (the recorded checksum is still checked by "
            "tests/integration/test_release.py::test_recorded_sha256_is_a_valid_hex_digest)"
        )

    assert _sha256(wheel) == recorded, (
        f"the SHA-256 of {wheel} does not match the one recorded in {MANIFEST_NAME}; "
        "the wheel was rebuilt or modified after the manifest was written"
    )
    assert wheel.stat().st_size == artifact.get("wheel_size_bytes"), (
        "the size of the wheel on disk does not match the recorded size"
    )


def test_recorded_sha256_is_a_valid_hex_digest(manifest):
    recorded = manifest.get("artifact", {}).get("wheel_sha256")
    assert recorded is not None
    assert len(recorded) == 64, "a SHA-256 digest is 64 hex characters"
    assert all(c in "0123456789abcdef" for c in recorded.lower())


def test_recorded_reproduction_command_is_not_empty(manifest):
    reproduction = manifest.get("reproduction", {})
    command = reproduction.get("command")
    assert command, "the manifest must record a reproduction command"
    assert isinstance(command, str)
    assert command.strip(), "the reproduction command must not be blank"
    assert reproduction.get("verified") is True, (
        "the recorded reproduction command must have been executed against the "
        "installed CLI"
    )
