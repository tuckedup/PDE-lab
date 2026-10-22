"""Generate the frozen v2 main-matrix experiment configurations.

The physics penalty weight comes from the v2 two-arm selection recorded
in `configs/lambda_selection.json` (Arm A - the required physics
comparison). It is read from that file, never hard-coded, so the matrix
and the selection cannot drift apart.

The cohort binds to explicit run ids, so a screening run that exactly
satisfies the protocol is reused rather than repeated.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
OUT = ROOT / "configs" / "experiments"
SEL = json.loads((ROOT / "configs" / "lambda_selection.json").read_text())

TEMPLATE = """# {title}
#
# Frozen v2 main-matrix row. The physics weight is the Arm-A selection
# from configs/lambda_selection.json (validation split only).
name: {name}
system: {system}
condition: {condition}
seed: {seed}
observation_dt: {dt}
lambda_pde: {lam}
tags: [{tags}]

model:
  architecture: {arch}
  history: 1
{model_extra}

data:
  dataset_dir: data/{system}

train:
  epochs: {epochs}
  batch_size: {bs}
  lr: 1.0e-3
  lr_schedule: cosine
  warmup_steps: 100
  grad_clip: 1.0
  weight_decay: 0.0
  device: auto
  max_walltime_minutes: 45

profiles:
  smoke:
    epochs: 2
    batch_size: {bs}
    max_walltime_minutes: 5
  prototype:
    epochs: {epochs}
    batch_size: {bs}
    max_walltime_minutes: 45
"""

FNO1 = "  modes: 16\n  width: 16\n  n_layers: 4"
FNO2 = "  modes_x: 8\n  modes_y: 8\n  width: 16\n  n_layers: 4"
UN1 = "  width: 16\n  n_levels: 3"
UN2 = "  width: 16\n  n_levels: 3"

DTS = {"burgers": 0.005, "allen_cahn": 0.1}
EPOCHS = {"burgers": 60, "allen_cahn": 40}
BS = {"burgers": 8, "allen_cahn": 2}

SYSTEMS = [
    ("burgers", "fno1d", FNO1),
    ("burgers", "unet1d", UN1),
    ("allen_cahn", "fno2d", FNO2),
    ("allen_cahn", "unet2d", UN2),
]

written = []
for system, arch, extra in SYSTEMS:
    lam = float(SEL[system]["arm_a_selected_lambda"])
    for cond in ("data_only", "physics"):
        for seed in (0, 1, 2):
            use_lam = 0.0 if cond == "data_only" else lam
            name = f"{system}_{arch}_{cond}_seed{seed}"
            tags = "main,v2" if not (cond == "physics" and arch.startswith("unet")) \
                else "main,optional_extension,v2"
            content = TEMPLATE.format(
                title=f"{system} / {arch} / {cond} (seed {seed}, lambda={use_lam})",
                name=name, system=system, condition=cond, seed=seed,
                dt=DTS[system], lam=use_lam, arch=arch, model_extra=extra,
                bs=BS[system], epochs=EPOCHS[system], tags=tags,
            )
            (OUT / f"{name}.yaml").write_text(content, encoding="utf-8")
            written.append(name)

print("Arm-A lambda per system:",
      {s: SEL[s]["arm_a_selected_lambda"] for s in SEL})
print(f"wrote {len(written)} v2 matrix configs")
for n in written:
    print("  ", n)
