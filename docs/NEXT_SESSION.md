# Next session

**State at the close of the v2-recovery campaign (2026-10-22).**

Everything below refers to the **v2** protocol
(`configs/protocol_v2.yaml`). The prototype results are withdrawn; see
`docs/LEGACY_INVALIDATION.md`.

## What is done and valid

* The reference solvers are repaired and validated: **19/19** Burgers and
  **14/14** Allen-Cahn checks, including manufactured solutions driven
  through the real integrator, sign-aware flux tests, exact
  observation-time tests, and reference own-error bounds at the real
  horizons.
* The dataset is regenerated with **exact** observation times and a
  **fresh, unseen test set** (IC seeds 7001/7002, zero overlap with the
  development set). Verify with
  `scripts/linux/verify_v2_data.py`.
* The **18 required main-matrix rows** are trained (3 seeds x 2 PDEs x 3
  conditions) and evaluated **once** on the fresh test set.
* The report is cohort-based, artifact-generated and reports every
  condition and seed separately (`docs/RESULTS.md`).
* A byte-reproducible wheel installs and runs from **outside** the source
  tree (`RELEASE_MANIFEST.json`, `docs/RELEASE.md`).
* 184 tests pass; lint is clean.

## Resource state

`campaigns/v2-recovery/BUDGET.json` holds the reconstructed spend:
heavy numerical **5.33 h of 6 h**; walltime **8.36 h against the 8 h
cap** (a 0.36 h overrun during the release/documentation phase, after
the matrix was frozen - recorded, not hidden); 20 training attempts of
30.

## The results, in one paragraph

All six declared targets are met on the frozen cohort, but the
scientific picture is qualified and the qualifications matter: the
**learned operators do not beat direct numerical stepping** (18-46x
worse on Burgers), the **Burgers residual gain is evaluator-specific**
(+70% on the declared primary implicit-Euler residual, **-45% to -63%**
on the independent centred first-derivative residual, under which the
*reference* is more PDE-consistent than either model), and the physics
arm **costs rollout accuracy** on both PDEs, so the overall method
recommendation is data-only.

## The next thing to do (in priority order)

1. **Investigate the Burgers residual discrepancy.** This is the most
   scientifically interesting open question. The physics arm optimises
   the implicit-Euler time map it is handed, and that map is *not* the
   continuous PDE. Hypotheses, in order of plausibility:
   * the penalty's implicit-Euler bias is the mechanism - test by
     evaluating the *reference* trajectory under the same discrete map
     and comparing the signed residual, not just the magnitude;
   * the penalty is over-regularising the small-amplitude early steps -
     test with a declared lambda warm-up (a *controlled* experiment,
     one variable);
   * the centred evaluator has a different support/truncation error on
     this system - test by measuring the reference's own centred
     residual under temporal refinement (the "floor" must be measured,
     not assumed).
   Any of these must be run on the **validation** split; the fresh test
   set has been inspected and must not be used to choose a change.

2. **Add a numerical baseline to the headline table** - it is already in
   the per-condition table, but the narrative should lead with it, since
   it is the honest context for the accuracy numbers.

3. **Near parameter-matched U-Net.** The width-4 U-Net is 0.63x / 0.51x
   the FNO's parameter count, not matched. Run a width search to land
   inside a declared tolerance (e.g. 10%), and report the mismatch if
   it cannot be reached.

4. **Optional rows that were not executed** (configs exist, runs do
   not): the physics-regularised U-Net at 3 seeds; the lambda grid at
   seeds 1 and 2 for the non-selected candidate; the width / spectral
   mode ablation on the repaired FNO2D; the maximum-throughput-at-best-
   batch benchmark profile; a parameter-OOD benchmark.

5. **Re-open the reference own-error budget.** The audit asked for the
   reference discrepancy to be at least an order of magnitude below the
   claimed model error. Measured reference own error: 2.2e-4 (Burgers)
   and 9.6e-8 (Allen-Cahn) at the old horizons; re-measure at the v2
   horizons and state the ratio to the model error explicitly.

## Exact commands

```bash
cd /home/atulp/projects/pdelab

# the numerical gate
.venv/bin/python scripts/linux/validate_solvers.py

# the tests
.venv/bin/python -m pytest tests/ -q

# verify the dataset identities (exact times, disjoint test ICs)
.venv/bin/python scripts/linux/verify_v2_data.py

# evaluate the frozen cohort on the fresh test set (once)
bash scripts/linux/evaluate_frozen_test.sh

# regenerate the report from the artifacts
.venv/bin/python -m pdelab.cli report

# the matched inference benchmark
.venv/bin/python -m pdelab.cli benchmark \
    --run-id burgers_fno1d_data_only_seed0 --profile matched

# the release
bash scripts/linux/release_all.sh

# the campaign state
.venv/bin/python scripts/linux/reconstruct_campaign_state.py
```

## Do not

* Do not re-tune anything against the fresh test set. It has been
  inspected; a future protocol (v3) with new test ICs is required for
  that.
* Do not present the cohort mean as a single method's score; use the
  per-condition and per-seed table.
* Do not describe the 96% memory figure as a device-memory saving; peak
  *reserved* memory rises 9% and total device memory is unchanged.
* Do not describe the Burgers residual gain as general physics
  superiority; it is evaluator-specific, and Allen-Cahn behaves
  differently.
* Do not backdate work. The January-June 2026 dates in the original
  brief have no support in this repository's execution history; the
  verified dates are 2026-10-19 (prototype) and 2026-10-21/22 (v2).
