"""Generate every experiment configuration for the confirmatory study.

Sets produced:

1. **Confirmatory matrix** - 2 PDEs x 2 architectures x 2 conditions x
   3 seeds = 24 runs. The physics-regularized U-Net is the declared
   *optional* extension beyond the 18-run minimum matrix; the sweep
   planner labels it as such.
2. **Lambda grid** - the pre-declared grid {0, 0.01, 0.1} at seed 0.
   lambda = 0 is the data-only baseline (already a separate config) and
   lambda = 0.01 is the confirmatory value, so only lambda = 0.1 needs
   its own config.
3. **Width / spectral-mode ablation** - the FNO at a smaller and a
   larger width+modes setting, at a fixed budget, seed 0.

The physics lambda is written from a single place (``LAMBDA``) so that
the selection made on the *validation* split propagates consistently to
every seed.
"""

import json
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
OUT = ROOT / "configs/experiments"

# ---------------------------------------------------------------------------
# The physics penalty weight used by the confirmatory matrix. Chosen on
# the validation split only (see scripts/linux/select_lambda.py).
# ---------------------------------------------------------------------------
SELECTION_PATH = ROOT / "configs" / "lambda_selection.json"
if SELECTION_PATH.exists():
    selection = json.loads(SELECTION_PATH.read_text())
    LAMBDA = {
        "burgers": float(selection["burgers"]["selected_lambda"]),
        "allen_cahn": float(selection["allen_cahn"]["selected_lambda"]),
    }
else:
    LAMBDA = {"burgers": 0.01, "allen_cahn": 0.01}
    print("no lambda selection found; using the pre-declared default 0.01")

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

DTS = {"burgers": 0.005, "allen_cahn": 0.1}
EPOCHS = {"burgers": 60, "allen_cahn": 40}
BS = {"burgers": 8, "allen_cahn": 2}
WALL = {"burgers": 45, "allen_cahn": 45}

SYSTEMS = [
    ("burgers", "fno1d", FNO1, "physics"),
    ("burgers", "unet1d", UN1, "physics"),
    ("allen_cahn", "fno2d", FNO2, "physics"),
    ("allen_cahn", "unet2d", UN2, "physics"),
]


written: list[str] = []
skipped: list[str] = []
RUNS = ROOT / "runs"


def write(name, system, arch, cond, seed, lam, extra, tags, epochs=None):
    """Write a config unless the run it describes already exists.

    A run whose artifacts already exist keeps its original config file,
    so the config hash recorded in its manifest stays valid. The
    generator is therefore idempotent and safe to re-run.
    """
    existing = RUNS / name
    if (existing / "manifest.json").exists() and (OUT / f"{name}.yaml").exists():
        skipped.append(name)
        return name
    content = TEMPLATE.format(
        title=f"{system} / {arch} / {cond} (seed {seed}, lambda={lam})",
        name=name, system=system, condition=cond, seed=seed,
        dt=DTS[system], lam=lam, arch=arch, model_extra=extra,
        bs=BS[system], epochs=epochs or EPOCHS[system], warmup=100,
        walltime=WALL[system], tags=tags,
    )
    (OUT / f"{name}.yaml").write_text(content, encoding="utf-8")
    written.append(name)
    return name


written: list[str] = []

# ---- 1. confirmatory matrix (24 runs) ----
for system, arch, extra, _ in SYSTEMS:
    for cond in ("data_only", "physics"):
        for seed in (0, 1, 2):
            lam = 0.0 if cond == "data_only" else LAMBDA[system]
            tags = "main"
            if cond == "physics" and arch.startswith("unet"):
                tags = "main,optional_extension"
            name = f"{system}_{arch}_{cond}_seed{seed}"
            write(name, system, arch, cond, seed, lam, extra, tags)
# ---- 2. lambda grid (lambda = 0.1 at seed 0, FNO only) ----
for system, arch, extra, _ in SYSTEMS:
    if not arch.startswith("fno"):
        continue
    name = f"{system}_{arch}_physics_lam0p1_seed0"
    write(name, system, arch, "physics", 0, 0.1, extra, "ablation,lambda_grid")

# ---- 3. width / spectral-mode ablation (seed 0, data-only FNO) ----
ABL = [
    ("burgers", "fno1d", "  modes: 8\n  width: 8\n  n_layers: 4", "w8m8"),
    ("burgers", "fno1d", "  modes: 24\n  width: 24\n  n_layers: 4", "w24m24"),
]
for system, arch, extra, tag in ABL:
    name = f"{system}_{arch}_{tag}_data_only_seed0"
    write(name, system, arch, "data_only", 0, 0.0, extra, "ablation,width_modes")

print(f"lambda used for the confirmatory matrix: {LAMBDA}")
print(f"wrote {len(written)} new configs, kept {len(skipped)} existing ones")
for n in written:
    print("  new:", n)
for n in skipped:
    print("  kept:", n)
