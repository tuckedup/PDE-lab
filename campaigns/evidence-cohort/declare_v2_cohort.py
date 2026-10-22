"""Declare the v2 cohort: dataset identities, the Arm-A lambda, the
U-Net members, and the current evaluator identity.

The cohort is the *frozen matrix*. It is written once, before the fresh
test set is evaluated, and the report then derives every target from
exactly the members declared here.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
sys.path.insert(0, str(ROOT / "src"))

from pdelab.evaluation.metrics import evaluator_identity  # noqa: E402

COHORT = ROOT / "campaigns" / "evidence-cohort" / "cohort.yaml"
SEL = json.loads((ROOT / "configs" / "lambda_selection.json").read_text())


def dataset_hashes() -> dict[str, dict[str, str]]:
    out = {}
    for system in ("burgers", "allen_cahn"):
        man = json.loads(
            (ROOT / f"data/{system}/{system}_manifest.json").read_text()
        )
        out[system] = {
            "system": system,
            "train_sha256": man["files"]["train"]["sha256"],
            "val_sha256": man["files"]["val"]["sha256"],
            "test_sha256": man["files"]["test"]["sha256"],
        }
    return out


def main() -> int:
    ds = dataset_hashes()
    ident = evaluator_identity(ROOT)
    text = COHORT.read_text()

    # dataset hashes
    for system, d in ds.items():
        for key in ("train_sha256", "val_sha256", "test_sha256"):
            text = re.sub(
                rf"({system}:.*?{key}: )\"?[0-9a-f]{{64}}\"?",
                lambda m, _d=d, _k=key: m.group(1) + _d[_k],
                text,
                flags=re.S,
            )

    # the Arm-A selected lambda for each physics member
    for system, arch in (("burgers", "fno1d"), ("allen_cahn", "fno2d")):
        lam = SEL[system]["arm_a_selected_lambda"]
        text = re.sub(
            rf"(- id: {system}_{arch}_physics\n(?:.*\n)*?    lambda_pde: )[0-9.]+",
            lambda m, _l=lam: m.group(1) + str(_l),
            text,
        )

    # evaluator identity
    text = re.sub(
        r'(evaluator:\n  id: pdelab-evaluation/\d+\n  sha256: ")[0-9a-f]+(")',
        lambda m: m.group(1) + ident["sha256"] + m.group(2),
        text,
    )

    # add the data-only U-Net members (the third required condition)
    if "burgers_unet1d_data_only" not in text:
        addition = """
  - id: burgers_unet1d_data_only
    role: baseline
    system: burgers
    architecture: unet1d
    condition: data_only
    lambda_pde: 0.0
    seeds: [0, 1, 2]
    run_ids:
      - burgers_unet1d_data_only_seed0
      - burgers_unet1d_data_only_seed1
      - burgers_unet1d_data_only_seed2

  - id: allen_cahn_unet2d_data_only
    role: baseline
    system: allen_cahn
    architecture: unet2d
    condition: data_only
    lambda_pde: 0.0
    seeds: [0, 1, 2]
    run_ids:
      - allen_cahn_unet2d_data_only_seed0
      - allen_cahn_unet2d_data_only_seed1
      - allen_cahn_unet2d_data_only_seed2

targets:"""
        text = text.replace("\ntargets:", addition, 1)

    # declare the U-Net as a member of the two accuracy targets
    text = text.replace(
        """  - name: burgers_nrmse
    metric: nrmse
    members:
      - burgers_fno1d_data_only
      - burgers_fno1d_physics""",
        """  - name: burgers_nrmse
    metric: nrmse
    members:
      - burgers_fno1d_data_only
      - burgers_fno1d_physics
      - burgers_unet1d_data_only""",
    )
    text = text.replace(
        """  - name: allen_cahn_nrmse
    metric: nrmse
    members:
      - allen_cahn_fno2d_data_only
      - allen_cahn_fno2d_physics""",
        """  - name: allen_cahn_nrmse
    metric: nrmse
    members:
      - allen_cahn_fno2d_data_only
      - allen_cahn_fno2d_physics
      - allen_cahn_unet2d_data_only""",
    )

    COHORT.write_text(text)
    print("cohort updated:")
    for system, d in ds.items():
        print(f"  {system}: train={d['train_sha256'][:12]}... "
              f"val={d['val_sha256'][:12]}... test={d['test_sha256'][:12]}...")
    print(f"  evaluator sha256: {ident['sha256']}")
    print("  Arm-A lambda: " + str({s: SEL[s]['arm_a_selected_lambda'] for s in SEL}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
