# PDELab - Physics-Constrained Neural Operator

A reproducible local PyTorch research prototype comparing **Fourier Neural
Operators** and **U-Nets**, with and without a **physics residual penalty**,
on two explicitly defined periodic PDE benchmarks.

* Package: `pdelab`  ·  Display title: *Physics-Constrained Neural Operator*
* Author: **Atul Pandey**
* Runtime: Ubuntu 24.04 on WSL2, NVIDIA RTX 4060 (8 GiB), PyTorch 2.14.1+cu130
* Protocol: **v2** (`configs/protocol_v2.yaml`)

> **Status: a corrected, 3-seed confirmatory study on a fresh held-out
> test set.** The prototype results reported before 2026-10-21 are
> **withdrawn** - they used a mis-timed dataset, a residual diagnostic
> that differentiated time twice, a sign-blind Burgers flux split, an
> FNO2D operator that dropped a spectral block, and a report that
> selected the best test run. See
> [`docs/LEGACY_INVALIDATION.md`](docs/LEGACY_INVALIDATION.md).

---

## The two benchmarks

| | System A | System B |
|---|---|---|
| equation | `u_t + (u^2/2)_x = nu u_xx` | `u_t = D (u_xx + u_yy) + a (u - u^3)` |
| name | 1D viscous Burgers | 2D diffusion-reaction (**Allen-Cahn**) |
| domain | periodic `[0, 1)` | periodic unit square |
| parameters | `nu = 0.01` | `D = 0.001`, `a = 1` |
| reference grid | 256 | 64 x 64 |
| observation grid | 128 | 32 x 32 |
| horizon | `t in [0, 0.5]`, `dt = 0.005` | `t in [0, 3.0]`, `dt = 0.1` |
| persistence baseline | 1.471 | 0.451 |

System B is explicitly the **scalar Allen-Cahn** system. It is not
Gray-Scott, and no two-species dataset is involved.

## Results - fresh test set, 3 seeds, frozen cohort

Free-rollout nRMSE on a **fresh, unseen** test set (IC seeds 7001/7002,
zero overlap with the development set):

| system | data-only FNO | physics FNO | data-only U-Net | numerical stepping |
|---|---|---|---|---|
| Burgers | **0.0156** | 0.0239 | 0.0165 | 0.00054 |
| Allen-Cahn | **0.0138** | 0.0160 | 0.0490 | 0.00275 |

| target | threshold | measured | status |
|---|---|---|---|
| burgers free-rollout nRMSE | < 0.05 | 0.0186 (cohort) | met |
| allen_cahn free-rollout nRMSE | < 0.20 | 0.0263 (cohort) | met |
| burgers primary residual reduction | >= 10% | +69.9% | met |
| allen_cahn primary residual reduction | >= 10% | +22.6% | met |
| same-checkpoint throughput ratio | >= 1.3 | 5.74x | met |
| same-workload peak **allocated** memory reduction | >= 20% | 96.05% | met |

`measured (cohort)` is the mean over the declared cohort members, **not**
the score of any single method. Per-condition and per-seed values are in
[`docs/RESULTS.md`](docs/RESULTS.md).

### What the results do and do not show

* **The learned operators do not beat direct numerical stepping** on
  either benchmark (18-46x worse on Burgers, 4.7-19x on Allen-Cahn).
* **The Burgers residual gain is evaluator-specific**: the declared
  primary (implicit-Euler) residual improves ~70% per seed, but the
  independent centred first-derivative residual is **45-63% worse**, and
  the reference solution is more PDE-consistent than either model under
  that evaluator. Allen-Cahn improves under both.
* **The physics arm costs rollout accuracy** on both PDEs, so the overall
  method recommendation under the global accuracy constraint is
  **data-only**. The physics arm remains the prespecified scientific
  comparison.
* The 96% memory figure is a **peak-allocated** figure; peak *reserved*
  memory rises 9% and total device memory is unchanged.

All numbers come from artifacts under `runs/`; none is typed by hand.

## Quick start

```bash
cd /home/atulp/projects/pdelab
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .

.venv/bin/python -m pdelab.cli doctor
.venv/bin/python scripts/linux/validate_solvers.py     # the phase gate
.venv/bin/python -m pdelab.cli data generate --config configs/pdes/burgers.yaml --profile prototype
.venv/bin/python -m pdelab.cli train --config configs/experiments/burgers_fno1d_data_only_seed0.yaml --profile prototype
.venv/bin/python -m pdelab.cli evaluate --run-id burgers_fno1d_data_only_seed0
.venv/bin/python -m pdelab.cli benchmark --run-id burgers_fno1d_data_only_seed0 --profile matched
.venv/bin/python -m pdelab.cli report
```

From Windows PowerShell:

```powershell
wsl -d Ubuntu --cd /home/atulp/projects/pdelab -- .venv/bin/python -m pdelab.cli doctor
```

## CLI

| command | purpose |
|---|---|
| `pdelab doctor` | environment and toolchain check |
| `pdelab resources check` | resource preflight against the hardware profile |
| `pdelab data generate --config <pde.yaml> --profile <p>` | generate a dataset |
| `pdelab solvers validate --system <burgers\|allen_cahn>` | numerical validation suite |
| `pdelab train --config <exp.yaml> --profile <p>` | train one condition |
| `pdelab evaluate --run-id <id>` | evaluate on the held-out test split |
| `pdelab sweep --config <matrix.yaml> --dry-run` | plan a run matrix |
| `pdelab benchmark --run-id <id> --profile <matched\|throughput>` | matched inference benchmark |
| `pdelab report` | build the result tables and plots from artifacts |

Profiles: `smoke` (tiny CPU checks), `prototype` (bounded one-seed
pilots), `full` (the frozen three-seed matrix, explicitly requested).

## Layout

```
src/pdelab/
  solvers/     validated reference solvers + the numerical validation suite
  data/        generation, HDF5 storage, splits, normalization
  models/      FNO 1D/2D, periodic U-Net 1D/2D, registry
  physics/     periodic operators, residual definitions, composite loss
  training/    trainer, experiment orchestration
  evaluation/  rollout, metrics, evaluation
  benchmarks/  process-isolated inference benchmark
  resources/   hardware profile, preflight, job lock
  reporting/   generated tables and plots
configs/       hardware, pdes, models, experiments, targets.yaml, prereg.yaml
tests/         unit, numerical, integration
scripts/       linux/ (runnable helpers), windows/ (PowerShell helpers)
docs/          ENVIRONMENT, PDE_SPEC, SOLVERS, DATA, DECISIONS, RESULTS,
               REPRODUCE, STATUS, NEXT_SESSION
```

## Honesty rules followed by this project

* The reference solvers are validated against analytic, manufactured,
  refinement and cross-solver tests **before** any data is generated.
* The reference solution is an *approximation*: its own measured error
  is reported (2.2e-4 for Burgers, 9.6e-8 for Allen-Cahn).
* The trivial persistence baseline is reported next to every accuracy
  number, so the difficulty of the task is visible.
* The PDE residual is reported with the reference's own residual under
  the identical evaluator, and with a separately declared second-order
  evaluator as a diagnostic.
* Inference comparisons use the same checkpoint, grid, horizon, batch
  size and output contract, and each variant is measured in a separate
  process.
* A missed target is reported as missed.
* One seed is one seed: no uncertainty is claimed from single runs.

## Citation

If you use this code, please cite the underlying papers rather than
this repository. See [`CITATIONS.md`](CITATIONS.md).

## License

Not yet declared. Add a `LICENSE` file before any redistribution.
