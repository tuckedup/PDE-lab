# PDELab - STATUS

**Campaign:** v2-recovery, 2026-10-21/22.
**Owner:** Atul Pandey.
**Protocol:** `configs/protocol_v2.yaml` (v2). The original preregistration is
preserved unchanged in `configs/prereg.yaml`.
**Result table:** `docs/RESULTS.md` (generated from artifacts, never hand-typed).

> The **prototype** results that were reported before 2026-10-21 are
> **withdrawn**: they were produced with a mis-timed dataset, a residual
> diagnostic that differentiated time twice, a sign-blind Burgers flux
> split, an FNO2D operator that dropped a spectral block, and a report
> that selected the best test run. See
> `docs/LEGACY_INVALIDATION.md`.

## 1. What is complete

| item | state | evidence |
|---|---|---|
| Environment (uv venv, torch 2.14.1+cu130, CUDA) | done | `pdelab doctor` |
| Resource profile, preflight, project lease on every heavy entrypoint | done | `pdelab resources check`; `campaigns/performance/check_project_lease.py` |
| 1D Burgers solver (WENO5-JS, global Lax-Friedrichs split) | done | 19/19 validation checks |
| 2D Allen-Cahn solver (spectral, exact diffusion, Strang) | done | 14/14 validation checks |
| Observation-time scheduling | done | exact to 1.7e-16; dataset manifest records the actual times |
| Corrected residual operators (primary implicit-Euler + independent centred) | done | `physics/operators.py` |
| FNO 1D/2D, periodic U-Net 1D/2D | done | 37 model tests |
| Cohort-based reporting with content identity and evidence states | done | `campaigns/evidence-cohort/cohort.yaml` |
| Benchmark correctness on full tensors | done | a sign flip and a spatial permutation are **rejected** |
| Fresh, unseen test set (IC seeds 7001/7002, zero dev overlap) | done | `scripts/linux/verify_v2_data.py` |
| **18 required main-matrix rows** (2 PDEs x 3 conditions x 3 seeds) | done | `campaigns/v2-recovery/MATRIX.json` |
| Matched inference benchmark, process-isolated, reset after warm-up | done | `runs/*/benchmark_matched.json` |
| Local release: wheel + clean install from outside the checkout | done | `RELEASE_MANIFEST.json`, `docs/RELEASE.md` |
| Test suite | done | **184 passed** |

## 2. Measured results - fresh test set, 3 seeds

Free-rollout nRMSE (the declared primary accuracy metric). The
**numerical stepping** column is a low-cost numerical baseline solved on
the same observation grid and horizon.

| system | condition | seeds | nRMSE mean | nRMSE SD | numerical stepping | persistence |
|---|---|---|---|---|---|---|
| burgers | FNO data-only | 0,1,2 | **0.0156** | 0.0011 | 0.00054 | 1.471 |
| burgers | FNO physics (λ=0.1) | 0,1,2 | **0.0239** | 0.0007 | 0.00054 | 1.471 |
| burgers | U-Net data-only | 0,1,2 | **0.0165** | 0.0071 | 0.00054 | 1.471 |
| allen_cahn | FNO data-only | 0,1,2 | **0.0138** | 0.0013 | 0.00275 | 0.451 |
| allen_cahn | FNO physics (λ=0.01) | 0,1,2 | **0.0160** | 0.0008 | 0.00275 | 0.451 |
| allen_cahn | U-Net data-only | 0,1,2 | **0.0490** | 0.0037 | 0.00275 | 0.451 |

Residual improvement of the physics arm over the matched data-only FNO,
**per seed**, under both evaluators:

| system | primary (implicit Euler) | independent (centred 1st derivative) |
|---|---|---|
| burgers | +69.53% / +69.20% / +70.97% | **−45.41% / −59.94% / −63.37%** |
| allen_cahn | +21.27% / +24.25% / +22.35% | +19.46% / +22.48% / +20.63% |

Inference, matched workload (same checkpoint, grid 128, 100 autoregressive
predictions, batch 1), each variant in a separate process:

| variant | median | peak allocated | peak reserved | correctness | joint pass |
|---|---|---|---|---|---|
| fp32 eager (baseline) | 0.260 s | 0.008346 GiB | 0.021484 GiB | - | - |
| autocast fp16 | 0.268 s | 0.008271 GiB | 0.021484 GiB | pass | no |
| torch.compile default | 0.309 s | 0.008298 GiB | 0.052734 GiB | pass | no |
| **torch.compile reduce-overhead** | **0.045 s** | **0.000329 GiB** | 0.023438 GiB | pass | **yes** |

## 3. Target status (frozen cohort, fresh test set)

| target | threshold | measured (cohort) | worst member | status |
|---|---|---|---|---|
| burgers free-rollout nRMSE | < 0.05 | 0.01864 | 0.02456 | **met** |
| allen_cahn free-rollout nRMSE | < 0.20 | 0.02628 | 0.05139 | **met** |
| burgers primary residual reduction | ≥ 10% | 69.90% | 69.20% | **met** |
| allen_cahn primary residual reduction | ≥ 10% | 22.62% | 21.27% | **met** |
| same-checkpoint throughput ratio | ≥ 1.3 | 5.736 | - | **met** |
| same-workload peak **allocated** memory reduction | ≥ 20% | 96.05% | - | **met** |

`measured (cohort)` is the mean over the declared cohort members, **not**
the score of any single method. The per-condition and per-seed values
are in `docs/RESULTS.md`.

## 4. The limitations that matter

These are the honest qualifications. They are not caveats bolted on; they
change what the results mean.

1. **The learned operators do not beat direct numerical stepping.**
   Burgers: numerical 0.00054 vs the learned models 0.0099-0.0246 (18-46x
   better). Allen-Cahn: numerical 0.00275 vs 0.0129-0.0514 (4.7-19x
   better). On these benchmarks the observation grids (128; 32x32) are
   close to the reference grids (256; 64x64) and the dynamics are smooth
   enough that a well-implemented solver on the observation grid is
   simply more accurate than a learned autoregressive operator. The
   operators are a *feasibility* result, not an accuracy win.

2. **The Burgers residual improvement is evaluator-specific.** The
   physics arm improves the declared primary (implicit-Euler) residual by
   about 70% per seed, but it makes the trajectory **45-63% worse** under
   the independent centred first-derivative residual - and under that
   evaluator the **reference solution is more PDE-consistent than either
   model** (reference 0.0413 vs data-only 0.101-0.112 vs physics
   0.163-0.167). The penalty optimises the implicit-Euler time map it
   is given; that is not the same as being more consistent with the
   continuous PDE. Allen-Cahn improves under both evaluators, so the
   effect is not universal in either direction.

3. **The physics arm costs rollout accuracy on both PDEs** (Burgers
   0.0156 -> 0.0239, Allen-Cahn 0.0138 -> 0.0160). Under the v2
   two-arm rule the **overall method recommendation is data-only** for
   both systems; the non-zero arm does not satisfy the global 5%
   accuracy constraint. The physics arm remains the *prespecified
   scientific comparison* and is reported as such.

4. **The memory result is about peak *allocated* memory, not device
   memory.** Peak allocated falls 96% (0.008346 -> 0.000329 GiB) because
   the CUDA-graph variant reuses a static pool instead of re-allocating
   the rollout intermediates every step. But peak **reserved** memory
   *rises* 9.1% (0.021484 -> 0.023438 GiB, the graph's private pool) and
   the device-level free-memory reading is essentially unchanged
   (6.8799 -> 6.8662 GiB, -0.2%). Calling this a device-memory saving
   would be wrong.

5. **The persistence baseline and the numerical-stepping baseline
   bracket the task.** Persistence is 1.471 (Burgers) and 0.451
   (Allen-Cahn), so the horizons are not trivial; but the numerical
   baseline is far better than the learned models.

## 5. Not executed

Declared but not run, because the campaign budget was spent on the
required matrix:

* the **physics-regularized U-Net** (optional extension; the configs
  exist, the rows were not run);
* the **λ grid at seeds 1 and 2** for the non-selected candidate (the
  screening was at seed 0 and the selected condition was then confirmed
  at three seeds, as the protocol prescribes);
* a **near parameter-matched U-Net** width search (the width-4 U-Net has
  12,409 parameters against the FNO's 19,825 in 1D and 35,377 against
  68,993 in 2D - i.e. 0.63x and 0.51x, not matched);
* the **width / spectral-mode ablation** on the repaired FNO2D;
* a **parameter-OOD or resolution-transfer** benchmark;
* the **maximum-throughput-at-best-batch** benchmark profile.

No hyperparameter was tuned against the fresh test set, and no optional
sweep was run to chase a headline.

## 6. Resources actually spent

Reconstructed from artifact timestamps and recorded walltimes
(`campaigns/v2-recovery/BUDGET.json`):

| quantity | spent | cap |
|---|---|---|
| heavy numerical job time | **5.33 h** (training 5.15, data 0.08, evaluation 0.07, benchmark 0.03) | 6 h |
| campaign wall time | **7.67 h** | 8 h |
| training attempts | 20 | 30 |
| peak CUDA allocated per run | 0.024 GiB | 4.5 GiB |
| peak host RSS per run | ~1.6 GiB | 3.5 GiB |
| dataset storage | 15 MiB | 4 GiB |

## 7. Legacy claims withdrawn

The prototype numbers (Burgers nRMSE 0.0372 "met", Allen-Cahn 0.0436
"met", +88% / +20.6% residual gains, 6.11x throughput, 1.72% memory)
are **withdrawn**. Reasons, each with reproduction evidence in
`campaigns/audit-verification/audit_reproduction.json`:

| prototype claim | why it is withdrawn |
|---|---|
| Burgers nRMSE 0.0372 "met" | the report selected the best test run; on the frozen cohort with 3 seeds the worst member is 0.02456 and the condition mean is 0.0239 |
| "second-order" residual gains | the operator computed `u_tt`, not `u_t`; every such value is invalid |
| +88% / +20.6% residual improvement | the primary metric is an implicit-Euler defect; on the independent centred evaluator the Burgers gain is negative |
| 6.11x throughput | measured on a mis-timed dataset and a repaired benchmark; re-measured as 5.74x on the corrected protocol |
| 1.72% memory reduction | measured with the reset before warm-up; re-measured with the corrected accounting (96% peak allocated, +9% reserved) |
| "all 89 tests passed" | the suite did not cover observation-time scheduling, the flux split, the FNO2D block coverage, the residual derivative, report selection or the benchmark correctness check; the suite now has 184 tests and the missing coverage is present |

## 8. Scope

A **local research prototype** of physics-regularized neural operators
on two self-defined periodic benchmarks, with validated reference
solvers, a frozen cohort, a fresh held-out test set and an
artifact-backed report. It is **not** a new neural-operator method, not
a comparison against published work, and not a claim of state of the
art. A local release is not a public deployment.
