# Legacy result invalidation

**Date:** 2026-10-21. This note records which prototype results are
withdrawn, why, and what replaces them. It is dated by the **actual**
audit and repair date; nothing here is backdated.

## Why

An independent audit of 2026-10-21 (three reviewers) found defects in
the prototype. I reproduced every finding against the checkout that
existed at that time, from the source itself, and stored the
reproduction in `campaigns/audit-verification/audit_reproduction.json`.
The findings were **real**: 6 of 8 reproduced exactly, and the two that
did not reproduce were detector artefacts (the benchmark-correctness
defect was real, but my string-based detector gave a false negative).

The defects and their consequences:

| # | defect | consequence |
|---|---|---|
| A | `solve_to` appended the first internal step as though it were the first positive observation, so requesting `[0, .1, .2, .3]` returned frames at `[0, .005, .1, .2]` | **every generated dataset was mis-timed**, and the physics residual used a `dt` that did not match the snapshot spacing |
| B | the "second-order" residual diagnostic computed `(nxt - 2*centre + prv)/dt**2`, i.e. `u_tt` | every reported "order2" residual value is invalid; the implemented quantity did not converge under refinement (0.03176 / 0.03146 / 0.03131) while the correct centred first derivative did (6.22e-6 / 1.56e-6 / 3.92e-7) |
| C | the Burgers flux split used the sign of `f = u^2/2`, so `f_minus` was identically zero and every state used a left-biased reconstruction | a perturbation on a negative background grew at **+0.88 per unit time** (antidiffusion) while the same perturbation on a positive background decayed at −0.88 |
| D | the FNO2D spectral operator retained only `out_ft[:, :, :mx, :my]`; `rfft2` keeps **both** signs of the first axis | an identity-weight operator had **93.6%** relative error against its own input and a single cosine had 50%: the negative-kx low-frequency block was never mixed. The architecture changed (68,993 -> 126,337 parameters), so the 2D FNO runs had to be retrained |
| E | the manufactured-solution check built the forcing from the same discrete RHS it then added back | it cancelled algebraically and passed to 4.8e-9 **without ever calling the integrator**; a wrong spatial operator would also have passed |
| F | the de-aliasing masks were not Hermitian-symmetric, the nonlinear output was never projected back to the retained band, and the cubic reaction used a *quadratic* 2/3 rule | the documented de-aliasing claim was false; the measured aliasing error was 1.05e-1 broadband and 5.0e-1 on the dataset IC distribution |
| G | the report picked the **minimum test nRMSE** across candidate runs to decide a target, and artifacts were bound only by a dataset hash | test-set selection; some `evaluation.json` artifacts were **older than the `best.pt`** they claimed to evaluate |
| H | the inference benchmark compared a **scalar checksum** (`mean(abs(output))`) | a sign flip or a spatial permutation would have passed the correctness gate |
| I | `pdelab benchmark` took neither the project lease nor the resource preflight | heavy jobs could overlap; the WSL virtual disk figure was reported as if it were physical free space |

The suite's 89 passing tests did **not** catch any of these. The
missing coverage is now present.

## Withdrawn prototype results

| prototype claim | value | status |
|---|---|---|
| Burgers free-rollout nRMSE | 0.0372, "met" | **withdrawn** - it was the best of several test runs. On the frozen 3-seed cohort the condition mean is 0.0239 and the worst member 0.02456; the *target* is still met, but the reported number was a selection artefact |
| Allen-Cahn free-rollout nRMSE | 0.0436, "met" | **withdrawn as a number** - the cohort mean is 0.0263 and the worst member 0.0514. The target is still met |
| Burgers residual improvement | +88.1% | **withdrawn** - re-measured as +69.9% (primary evaluator), and the gain is evaluator-specific |
| Allen-Cahn residual improvement | +20.6% | **withdrawn** - re-measured as +22.6% on the corrected dataset |
| "order2" residual values | +23.8% / −2.6% | **withdrawn** - the operator computed `u_tt` |
| inference throughput ratio | 6.11x | **withdrawn** - re-measured as 5.74x on the repaired benchmark and the corrected protocol |
| inference memory reduction | 1.72% | **withdrawn** - measured with the reset before warm-up; re-measured as 96.05% peak *allocated* with peak *reserved* up 9.09% |
| "89 tests pass" | - | **withdrawn as evidence of validity** - the suite lacked the coverage for every defect above. The suite now has 184 tests including the missing coverage |
| "reference solution own error 2.2e-4 / 9.6e-8" | - | **withdrawn** - measured on a mis-timed generator. Re-measured at the real horizons over several random ICs |
| "parameter matching" for the width-4 U-Net | "not a capacity artefact" | **withdrawn as a matching claim** - 12,409 vs 19,825 (1D) and 35,377 vs 68,993 (2D) is 0.63x and 0.51x, not matched |

## What survives, and why

* The **engineering feasibility** of the study: training runs finish on
  the laptop, a physics-regularised FNO trains stably, a real
  parameter-gradient path exists for the PDE penalty, the evaluation and
  benchmark harnesses work, and a byte-reproducible local release
  installs and runs from outside the source tree.
* The **reference solvers**, after repair: 19/19 Burgers and 14/14
  Allen-Cahn checks pass, including manufactured solutions driven through
  the real integrator, sign-aware flux tests, exact observation-time
  tests and reference own-error bounds at the real horizons.
* The **methodological infrastructure**: a frozen cohort, content
  identities, evidence states, true resume, a hard physics-gradient
  gate, and a full-tensor benchmark correctness check.

## What replaces them

The v2 results in `docs/RESULTS.md` and `docs/STATUS.md`: 18 required
main-matrix rows on a **fresh, unseen test set** (IC seeds 7001/7002,
zero overlap with the development set), a two-arm lambda selection fixed
prospectively on the validation split, and a report that derives every
target from the declared cohort.

The v2 result is *different* from the prototype result, and in the
important places it is **worse**: the physics arm no longer improves
accuracy, the Burgers residual gain is evaluator-specific, and the
learned operators are beaten by direct numerical integration. Those are
the honest results.
