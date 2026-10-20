"""Near parameter-matched U-Net configurations.

The pilot U-Net (width 16) has 8-10x more parameters than the FNO, so
the comparison is generous to the spatial baseline. These narrower
variants bring the parameter counts close to the FNO's, which is the
documented near parameter-matched comparison. The practical-budget
comparison (width 16) is retained as the primary one.

Parameter matching does NOT equalise FLOPs or optimisation difficulty,
and the report states the counts so the reader can judge.
"""

from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab/configs/experiments")

TEMPLATE = """# {title}
#
# Near parameter-matched U-Net. The pilot width-16 U-Net has ~8-10x the
# parameters of the FNO; this variant brings the counts close. It is a
# controlled ablation at a fixed budget, not a replacement for the
# practical-budget comparison.
name: {name}
system: {system}
condition: data_only
seed: {seed}
observation_dt: {dt}
lambda_pde: 0.0
tags: [ablation, parameter_matched]

model:
  architecture: {arch}
  history: 1
  width: {width}
  n_levels: {levels}

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

SPECS = [
    ("burgers", "unet1d", 4, 3, 0.005, 8, 60),
    ("burgers", "unet1d", 6, 3, 0.005, 8, 60),
    ("allen_cahn", "unet2d", 4, 3, 0.1, 2, 40),
    ("allen_cahn", "unet2d", 6, 3, 0.1, 2, 40),
]

for system, arch, width, levels, dt, bs, epochs in SPECS:
    for seed in [0]:
        name = f"{system}_{arch}w{width}_data_only_seed{seed}"
        content = TEMPLATE.format(
            title=f"{system} / {arch} width {width} (near parameter-matched)",
            name=name, system=system, arch=arch, width=width, levels=levels,
            dt=dt, seed=seed, bs=bs, epochs=epochs,
        )
        (ROOT / f"{name}.yaml").write_text(content, encoding="utf-8")
        print("wrote", name)
