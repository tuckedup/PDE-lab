# Reproducing the results

Everything runs inside WSL2. The commands below are given in the form
they are executed.

## 0. One-time environment setup

```bash
cd /home/atulp/projects/pdelab
export PATH="$HOME/.local/bin:$PATH"     # uv lives here
uv venv --python 3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
```

From Windows PowerShell:

```powershell
wsl -d Ubuntu --cd /home/atulp/projects/pdelab -- pwd
wsl -d Ubuntu --cd /home/atulp/projects/pdelab -- .venv/bin/python -m pdelab.cli doctor
```

## 1. Environment and resource checks

```bash
.venv/bin/python -m pdelab.cli doctor
.venv/bin/python -m pdelab.cli resources check
```

`doctor` must report `cuda.available: true` and
`cuda.forward_backward_ok: true`. `resources check` must report
`"violations": []`.

## 2. Numerical validation (the phase gate)

```bash
.venv/bin/python scripts/linux/validate_solvers.py
```

Expected: `OVERALL: PASS`, 12/12 checks for each system. **Do not
generate data before this passes.**

## 3. Dataset generation

```bash
.venv/bin/python -m pdelab.cli data generate --config configs/pdes/burgers.yaml    --profile prototype
.venv/bin/python -m pdelab.cli data generate --config configs/pdes/allen_cahn.yaml --profile prototype
```

Measured cost: 34 s (burgers) and 207 s (allen_cahn) on this machine.
The `--profile smoke` variant generates a tiny dataset for a quick
check.

## 4. Training (one condition at a time)

```bash
.venv/bin/python -m pdelab.cli train --config configs/experiments/burgers_fno1d_data_only_seed0.yaml --profile prototype
.venv/bin/python -m pdelab.cli train --config configs/experiments/burgers_fno1d_physics_seed0.yaml    --profile prototype
.venv/bin/python -m pdelab.cli train --config configs/experiments/burgers_unet1d_data_only_seed0.yaml --profile prototype
.venv/bin/python -m pdelab.cli train --config configs/experiments/allen_cahn_fno2d_data_only_seed0.yaml  --profile prototype
.venv/bin/python -m pdelab.cli train --config configs/experiments/allen_cahn_fno2d_physics_seed0.yaml     --profile prototype
.venv/bin/python -m pdelab.cli train --config configs/experiments/allen_cahn_unet2d_data_only_seed0.yaml --profile prototype
```

Measured wall times: 635 s, 800 s, 1289 s, 516 s, 642 s, 1372 s.

The job lock refuses to start a second concurrent PDELab job. The
commands must be run **sequentially** and must not be run while another
heavy job is using the GPU.

## 5. Evaluation

```bash
.venv/bin/python scripts/linux/evaluate_all.sh
```

This evaluates every completed run that has no `evaluation.json` and
prints the comparison table. The individual command is:

```bash
.venv/bin/python -m pdelab.cli evaluate --run-id burgers_fno1d_physics_seed0
```

## 6. Inference benchmark

```bash
.venv/bin/python scripts/linux/benchmark_all.sh
```

Each variant is measured in a **separate process** so the CUDA caching
allocator cannot contaminate the peak-memory number. The GPU must be
otherwise idle; check `nvidia-smi` first and do not benchmark while
another project is using the device.

## 7. Report and plots

```bash
.venv/bin/python -m pdelab.cli report
```

This writes `results/RESULTS.md`, `results/results.json` and
`docs/RESULTS.md` from the stored artifacts, plus the plots in
`results/plots/`. **No number in the report is typed by hand.**

## 8. Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Expected: 86 passed. The suite runs on CPU in about 20 s and needs no
GPU.

## 9. The full (confirmatory) matrix - NOT executed yet

```bash
# dry run first: this prints the plan and the cost estimate, and runs nothing
.venv/bin/python -m pdelab.cli sweep --config configs/experiments/main_matrix.yaml --profile full --dry-run

# only after reviewing the estimate, and only with the machine otherwise idle
.venv/bin/python -m pdelab.cli sweep --config configs/experiments/main_matrix.yaml --profile full
```

The full matrix is 2 PDEs x 3 conditions x 3 seeds = 18 runs, plus the
optional physics-regularized U-Net (6 more), for 24 in total. At the
measured prototype cost (about 12 minutes per run) this is roughly
4.8 hours of sequential training (3.6 hours for the required 18), which
**exceeds the declared 120-minute sweep budget**. It must therefore be
split across sessions, or the budget must be raised explicitly first.

The dry run reports the split explicitly:

```
total runs             : 24
  required (minimum)   : 18  (2 PDEs x 3 conditions x 3 seeds = 18 runs)
  optional extension   : 6
estimated total minutes: 288
```

The three-seed matrix also needs the seed-1 and seed-2 experiment
configs, which are not yet written; the sweep runner reports a run as
`missing_config` rather than silently skipping it. Note also that the
sweep plan does **not** override the per-config batch size: the 2D
runs keep their declared `batch_size: 2`, which is a hardware limit.

## Continuation

Every run manifest records the exact continuation command and the
resolved configuration, the dataset hashes, the Git revision and the
failure state if anything went wrong. A run that was interrupted can be
resumed from its `last.pt` checkpoint, which stores the model,
optimizer, scheduler, epoch, global step and RNG states.

## Windows-side invocation

To drive the project from Windows without opening a shell inside WSL:

```powershell
wsl -d Ubuntu --cd /home/atulp/projects/pdelab -- .venv/bin/python -m pdelab.cli doctor
wsl -d Ubuntu --cd /home/atulp/projects/pdelab -- .venv/bin/python -m pdelab.cli report
```

For substantial work, prefer running the repository `.sh` scripts
through WSL rather than nesting complex quoting inside PowerShell.
