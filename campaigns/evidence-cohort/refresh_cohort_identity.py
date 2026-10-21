"""Refresh the declared evaluator identity in the cohort declaration.

The cohort pins the *content identity* of the evaluation
implementation (the SHA-256 of the per-file digests of
``src/pdelab/evaluation/{evaluate,metrics,rollout}.py`` and
``src/pdelab/physics/operators.py``).  Whenever one of those files
changes, the previously produced artifacts stop being comparable and
the cohort has to be re-declared with the new identity.

    python campaigns/evidence-cohort/refresh_cohort_identity.py

The script only rewrites the ``evaluator.sha256`` entry of
``campaigns/evidence-cohort/cohort.yaml`` and prints the per-file
digests.  It never touches the members, the seeds or the dataset
hashes: those are the *declared* cohort and are not derived from the
code.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pdelab.evaluation.metrics import evaluator_identity  # noqa: E402

COHORT = ROOT / "campaigns" / "evidence-cohort" / "cohort.yaml"


def main() -> int:
    identity = evaluator_identity(ROOT)
    text = COHORT.read_text()
    new = re.sub(
        r'(evaluator:\n  id: pdelab-evaluation/2\n  sha256: ")[0-9a-f]+(")',
        lambda m: m.group(1) + identity["sha256"] + m.group(2),
        text,
    )
    if new != text:
        COHORT.write_text(new)
        print("cohort updated")
    else:
        print("cohort already up to date")
    print("evaluator id   :", identity["id"])
    print("evaluator sha256:", identity["sha256"])
    for name, digest in identity["files"].items():
        print(f"   {name}: {digest}")
    print()
    print("Every artifact produced with a different evaluator identity is")
    print("reported as invalid_protocol until it is re-evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
