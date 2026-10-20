"""Generate the experiment configuration files.

Kept as a script so the matrix is reproducible and the shared settings
are guaranteed to be identical across conditions.
"""

from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab/configs/experiments")
ROOT.mkdir(parents=True, exist_ok=True)

TEMPLATE = """# {title}
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

train:
  epochs: {epochs}
  batch_size: {bs}
  lr: 1.0e-3
  lr_schedule: cosine
  warmup_steps: {warmup}
  grad_clip: 1.0
  weight_decay: 0.0
  device: auto
  max_walltime_minutes: {walltime}

profiles:
  smoke:
    epochs: 2
    batch_size: {bs}
    max_walltime_minutes: 5
  prototype:
    epochs: {epochs}
    batch_size: {bs}
    max_walltime_minutes: {walltime}
"""

FNO1 = "  modes: 16\n  width: 16\n  n_layers: 4"
FNO2 = "  modes_x: 8\n  modes_y: 8\n  width: 16\n  n_layers: 4"
UN1 = "  width: 16\n  n_levels: 3"
UN2 = "  width: 16\n  n_levels: 3"

SPECS = [
    ("burgers", "fno1d", "data_only", 0.0, FNO1, 8, 60, 100, 45, "baseline,main"),
    ("burgers", "fno1d", "physics", 0.01, FNO1, 8, 60, 100, 45, "physics,main"),
    ("burgers", "unet1d", "data_only", 0.0, UN1, 8, 60, 100, 45, "baseline,main"),
    ("burgers", "unet1d", "physics", 0.01, UN1, 8, 60, 100, 45, "physics,optional"),
    ("allen_cahn", "fno2d", "data_only", 0.0, FNO2, 2, 40, 100, 45, "baseline,main"),
    ("allen_cahn", "fno2d", "physics", 0.01, FNO2, 2, 40, 100, 45, "physics,main"),
    ("allen_cahn", "unet2d", "data_only", 0.0, UN2, 2, 40, 100, 45, "baseline,main"),
    ("allen_cahn", "unet2d", "physics", 0.01, UN2, 2, 40, 100, 45, "physics,optional"),
]

DTS = {"burgers": 0.005, "allen_cahn": 0.05}

for system, arch, cond, lam, extra, bs, epochs, warmup, walltime, tags in SPECS:
    for seed in [0]:
        name = f"{system}_{arch}_{cond}_seed{seed}"
        content = TEMPLATE.format(
            title=f"{system} / {arch} / {cond} (seed {seed})",
            name=name,
            system=system,
            condition=cond,
            seed=seed,
            dt=DTS[system],
            lam=lam,
            arch=arch,
            model_extra=extra,
            bs=bs,
            epochs=epochs,
            warmup=warmup,
            walltime=walltime,
            tags=tags,
        )
        (ROOT / f"{name}.yaml").write_text(content, encoding="utf-8")
        print("wrote", name)
