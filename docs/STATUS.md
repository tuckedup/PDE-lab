# Status

**Date of this snapshot:** 2026-10-20 (UTC)
**Project:** PDELab - Physics-Constrained Neural Operator
**Owner:** Atul Pandey

Everything below was executed on this machine. Nothing is projected.

---

## 1. Implemented and verified

| component | state | evidence |
|---|---|---|
| Environment (uv venv, PyTorch 2.14.1+cu130, CUDA) | done | `pdelab doctor`: CUDA available, forward/backward ok |
| Resource profile + preflight + project job lock | done | `pdelab resources check`: no violations |
| 1D Burgers reference solver (WENO5-JS + SSPRK(3,3)) | done | 12/12 numerical checks pass |
| 2D Allen-Cahn reference solver (spectral + Strang) | done | 12/12 numerical checks pass |
| Independent cross-check solvers (MUSCL, dealiased spectral) | done | agreement 9.4e-6 / 4.4e-4 |
| Deterministic dataset generation, HDF5 + manifests | done | 96 trajectories per system, reproduces to 6e-8 |
| FNO 1D/2D (learned complex spectral mixing) | done | shape / mode / gradient tests |
| Periodic U-Net 1D/2D (circular padding, nearest-upsample) | done | periodicity + covariance tests |
| Physics residual operators + composite loss | done | residual-gradient check passes in every physics run |
| One-step and free-rollout evaluation, nRMSE + residual metrics | done | `evaluation.json` per run |
| Matched inference benchmark, process-isolated variants | done | `benchmark_matched.json` per run |
| Test suite | done | **86 tests pass** |
| Plots and generated result tables | done | `results/plots/`, `docs/RESULTS.md` |
| Documentation set | done | `docs/*.md` |

## 2. Measured results (prototype profile, **1 seed**, RTX 4060)

| run | params | rollout nRMSE | 1-step nRMSE | persistence | residual | reference residual | train s |
|---|---|---|---|---|---|---|---|
| burgers FNO data-only | 19,825 | **0.0813** | 0.00352 | 1.478 | 1.006 | 1.084 | 635 |
| burgers FNO physics (λ=0.01) | 19,825 | **0.0372** | 0.00275 | 1.478 | **0.120** | 1.084 | 800 |
| burgers U-Net data-only (width 16) | 191,329 | **0.0951** | 0.00398 | 1.478 | 2.246 | 1.084 | 1291 |
| burgers U-Net data-only (width 4, matched) | 12,409 | **0.6831** | 0.2868 | 1.478 | 6.007 | 1.084 | 1292 |
| allen_cahn FNO data-only | 68,993 | **0.0443** | 0.00530 | 0.415 | 0.0815 | 0.0607 | 516 |
| allen_cahn FNO physics (λ=0.01) | 68,993 | **0.0436** | 0.00509 | 0.415 | **0.0647** | 0.0607 | 642 |
| allen_cahn U-Net data-only (width 16) | 558,529 | **0.0457** | 0.00788 | 0.415 | 0.0424 | 0.0607 | 1333 |
| allen_cahn U-Net data-only (width 4, matched) | 35,377 | **0.1325** | 0.07313 | 0.415 | 0.1603 | 0.0607 | 1296 |

Relative residual improvement of the physics FNO over the matched
data-only FNO: **+88.1%** (Burgers) and **+20.6%** (Allen-Cahn) under
the primary evaluator; **+23.8%** and **-2.6%** under the separately
declared second-order diagnostic evaluator.

### The near parameter-matched comparison

The pilot U-Net (width 16) has 8-10x the FNO's parameters. A near
parameter-matched U-Net (width 4) was therefore also trained:

| system | model | params | rollout nRMSE |
|---|---|---|---|
| burgers | FNO | 19,825 | **0.0813** |
| burgers | U-Net width 4 | 12,409 (0.63x) | 0.6831 |
| allen_cahn | FNO | 68,993 | **0.0443** |
| allen_cahn | U-Net width 4 | 35,377 (0.51x) | 0.1325 |

The FNO is ahead at both parameter-matched sizes, and the narrow U-Net
is far behind (it overfits: its best validation epoch is early and its
validation loss is 2-3 orders of magnitude worse). So the FNO's
advantage in this study is **not** a parameter-count artefact.
Parameter matching does not equalise FLOPs or optimisation difficulty.

Inference, matched workload (same checkpoint, grid, horizon, batch
size), each variant in a separate process:

| variant | median (1D FNO) | peak CUDA allocated |
|---|---|---|
| fp32 eager (baseline) | 0.245 s | 0.00830 GiB |
| autocast fp16 | 0.233 s | 0.00822 GiB |
| torch.compile default | 0.275 s | 0.00825 GiB |
| torch.compile reduce-overhead (CUDA graphs) | **0.040 s** | 0.00816 GiB |

**Throughput ratio 6.11** (target ≥ 1.3). **Peak-memory reduction
1.72%** on the declared reference model (target ≥ 20%); the 2D U-Net
reaches 25.6% because its weights dominate the peak.

## 3. Target status

| target | threshold | measured | status |
|---|---|---|---|
| burgers held-out nRMSE | < 0.05 | 0.0372 (physics FNO) | **met** |
| allen_cahn held-out nRMSE | < 0.20 | 0.0436 (physics FNO) | **met** |
| burgers residual improvement | ≥ 10% | +88.1% | **met** |
| allen_cahn residual improvement | ≥ 10% | +20.6% | **met** |
| inference throughput ratio | ≥ 1.3 | 6.11 | **met** |
| inference peak-memory reduction | ≥ 20% | 1.72% | **missed** |

The burgers nRMSE target is met by the physics-regularized FNO but
**missed** by the data-only FNO (0.0813) and by the U-Net (0.0668).

## 4. NOT completed

These are declared requirements that are **not** yet met. They are
listed so that no part of this project is over-claimed.

1. **Three independent training seeds.** Only seed 0 has been run. The
   requirement applies to independent *training* runs, not to three
   inference repetitions of one checkpoint. All reported numbers are
   single-seed and carry no uncertainty estimate.
2. **The 18-run main matrix.** Not executed. The sweep is planned,
   costed and dry-run tested, but the seed-1 and seed-2 experiment
   configs are not yet written.
3. **Physics-regularized U-Net.** The condition is configured
   (`burgers_unet1d_physics_seed0.yaml`,
   `allen_cahn_unet2d_physics_seed0.yaml`) but not run.
4. **The λ grid {0, 0.01, 0.1}.** Only λ = 0 and λ = 0.01 have been
   run, and only at seed 0. A validation-only search has not been
   performed.
5. **Near parameter-matched U-Net.** **Done** at width 4 (see section
   2). A width-6 variant is configured but not run.
6. **One-step vs rollout horizon ablation** and the **spectral
   modes / width** ablation are configured in spirit but not executed.
   The one-step and rollout numbers are already reported separately
   for every run.
7. **The maximum-throughput-at-best-batch profile.** The `throughput`
   benchmark profile exists but has not been run and reported.
8. **Parameter-OOD / resolution-transfer benchmark.** Not implemented.

## 5. Bugs found by the validation suite and the tests

These were real defects in this implementation, found and fixed. They
are listed because they shaped the final design.

1. **Mis-indexed Burgers flux divergence** - a full-cell shift that
   left the scheme consistent but first-order and wrong.
2. **Mismatched SSPRK(3,3) stage arguments and weights** - the
   integrator was not even consistent with `du/dt = a u`.
3. **Mis-truncated 2D spectral restriction** - `rfft2` keeps both signs
   of the first axis, so a one-sided truncation halved every field.
4. **Broken FNO2D parameter conditioning** - an `expand` to four
   dimensions from a three-dimensional tensor.
5. **U-Net checkerboard artefact** from `ConvTranspose(kernel=2,
   stride=2)`, which broke the constant-field test.
6. **Wrong free-energy functional for Allen-Cahn** - the gradient term
   needs a `D/2` coefficient; with `1/2` the free energy appeared to
   increase, mimicking an instability.
7. **Allen-Cahn reference step chosen from the stability limit rather
   than an accuracy limit** - `dt = 0.5` is stable but far too coarse
   for a reference.
8. **The `restrict_2d` operator halved every field** - `rfft2` retains
   both signs of the first axis, so a one-sided row truncation removed
   the negative-kx modes. Caught by a unit test.
9. **`GroupNorm(min(8, channels), channels)`** - invalid for channel
   counts not divisible by 8, which broke every U-Net width other than
   8/16/32. Caught when building the width-4 variant.
10. **`run_id_for` ignored the configuration's `name`** - the
    near parameter-matched U-Net overwrote the practical-budget U-Net's
    run directory. Caught by noticing that the parameter count in the
    report did not match the configuration.
11. **GPU training is not bitwise deterministic.** The same 1D U-Net
    configuration, re-run at the same seed, changed its validation loss
    by ~20% and its held-out nRMSE by ~40%. cuDNN autotuning was the
    main cause; it is now disabled and deterministic kernels are
    selected, and the manifest records the determinism settings. The
    spread is a further reason the three-seed requirement matters.

## 6. Scope statement

This is a **local research prototype** of a physics-regularized neural
operator on two self-defined periodic benchmarks. The result is: adding
a PDE-residual penalty to an FNO measurably reduced the physical
residual and, on this prototype, also improved the rollout accuracy on
1D Burgers, while a matched-workload CUDA-graph inference pipeline gave
a 6.1x throughput gain.

It is **not** a claim of a new neural-operator method, and it is **not**
a comparison against published work. The two systems are explicit
periodic benchmark definitions, the reference solvers are validated
against analytic, manufactured and refinement tests, and the numbers
above come from single-seed runs on one GPU.
