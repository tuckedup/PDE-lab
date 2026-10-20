# Next session

Everything below is a concrete, executable next step. Nothing here has
been run.

## 0. Read first

* `docs/STATUS.md` - what is done, what is measured, what is not.
* `docs/DECISIONS.md` - why the configuration and the methods are what
  they are.
* `results/RESULTS.md` - the generated target table.

## 1. Immediate: finish the confirmatory matrix

The single largest gap is that only **seed 0** has been trained. The
requirement is three independent training seeds per condition.

### 1a. Write the seed-1 and seed-2 configs

```bash
.venv/bin/python scripts/linux/gen_experiment_configs.py    # currently seeds [0] only
```

Extend `scripts/linux/gen_experiment_configs.py` to emit
`{system}_{arch}_{condition}_seed{1,2}.yaml` for the six main
conditions (3 per system x 2 systems x 2 seeds = 12 new configs), and
add the physics-regularized U-Net configs, which already exist for
seed 0.

### 1b. Estimate the cost before running

```bash
.venv/bin/python -m pdelab.cli sweep --config configs/experiments/main_matrix.yaml --profile full --dry-run
```

At the measured prototype cost (~12 min/run) the 18-run required matrix
is about 3.6 hours (24 runs including the optional physics U-Net is
about 4.8 hours), which **exceeds the declared 120-minute sweep
budget**. Pick one:

* split the matrix across sessions (recommended - the budget is a
  standing limit, not a target to raise), or
* raise `max_default_sweep_walltime_minutes` in
  `configs/hardware/atul_laptop.yaml` **explicitly and in writing**, or
* reduce the epoch count for the confirmatory runs (this changes the
  protocol, so it must be recorded in `configs/prereg.yaml`).

### 1c. Run the matrix sequentially

```bash
.venv/bin/python -m pdelab.cli sweep --config configs/experiments/main_matrix.yaml --profile full
```

The job lock enforces one job at a time. Check `nvidia-smi` before
starting and do not run while another project is using the GPU.

## 2. Ablations that are declared but not run

The pre-registration lists these; they are the smallest justified set.

| ablation | how | note |
|---|---|---|
| no physics penalty vs the selected penalty | already measured at seed 0 | repeat over 3 seeds |
| residual weight λ ∈ {0, 0.01, 0.1} | 2 new configs per system | **validation split only**; the physics model must not be tuned against the test set |
| one-step vs rollout horizon | already reported separately per run | `nrmse_one_step_teacher_forced` vs `nrmse` |
| spectral modes / model width at a fixed budget | 2-3 extra configs | keep the parameter count fixed and report it |
| correctness of residual gradients | already automated | `physics_grad_check` in every manifest |
| near parameter-matched U-Net | width 4-6 U-Net configs | the U-Net is currently 8-10x larger than the FNO |

## 3. Two things worth fixing before the confirmatory runs

1. **The allen_cahn residual improvement is evaluator-dependent.**
   Under the primary (forward-difference) evaluator the physics model
   improves the residual by 20.6%, but under the second-order
   diagnostic evaluator it is 2.6% *worse*. The Burgers result is
   positive under both (88.1% / 23.8%). This should be investigated
   before the confirmatory matrix, because it may be a λ-scaling issue:
   the residual magnitudes differ by three orders of magnitude between
   the two systems (1.0 vs 0.08), so a single `pde_scale` may not suit
   both. The scales are currently frozen at 1.0 for both.

2. **The inference memory target is missed at the declared reference
   model** (1.7% against a 20% target) because the model is small
   enough that the cuFFT workspace dominates the peak. Either
   (a) report it as missed and explain, which is the current state, or
   (b) measure at a batch size where activations dominate and label
   that result clearly as a *different* workload, which the brief
   allows but requires to be labelled. The `throughput` benchmark
   profile (batch sweep 1/2/4/8) is implemented but has not been run.

## 4. Optional extension

A parameter-OOD or resolution-transfer benchmark must have **separate
labels** and must not replace the main held-out test. The infrastructure
needed is the PDE-parameter channel in the models (already implemented
and tested, but unused: `n_params` is 0 in every current config) and a
second dataset with different `nu` / `D`.

## 5. Housekeeping

* `git add -A && git commit` - the working tree has uncommitted changes
  since the last commit, and every future run manifest will record the
  revision and the dirty flag.
* Re-run `pdelab report` after any new run so `docs/RESULTS.md` stays
  generated rather than hand-written.
* Consider a `CITATIONS.md` entry for the two reference solvers and for
  the neural-operator and physics-informed-operator papers listed in
  `CITATIONS.md`.
