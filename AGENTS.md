# AGENTS.md - persistent instructions for this project

PDELab: a reproducible local PyTorch research prototype comparing
Fourier Neural Operators and U-Nets, with and without a physics
residual penalty, on two explicitly defined periodic PDE benchmarks.
Owner: **Atul Pandey**.

## 1. Execution boundary

* The agent (OpenCode) runs on **Windows**; all computation runs inside
  **Ubuntu WSL2**.
* Canonical project directory: `/home/atulp/projects/pdelab`
* Windows view: `\\wsl.localhost\Ubuntu\home\atulp\projects\pdelab`
* One repository, one Linux `.venv`. Never duplicate the project into a
  second Windows directory, and never write into `C:\Windows\System32`,
  the npm install folder, or the FlyCircuit repository.
* Use `wsl -d Ubuntu --cd /home/atulp/projects/pdelab -- <cmd>` for
  short commands. For anything substantial, put the logic in a
  repository `.sh`/`.py` file and run it through WSL. Nested Bash
  quoting inside PowerShell is unreliable (note: `|` and `>` are
  interpreted by PowerShell before WSL sees them) - prefer script
  files.
* Use Linux `uv`, Python and numerical libraries. Do not modify or use
  the Conda base environment. `uv` is at `~/.local/bin/uv` and is not on
  a fresh shell's PATH.
* Do not create `CLAUDE.md` as the main instruction file. This file is
  the instruction file.

## 2. Resource limits (conservative laptop profile)

`configs/hardware/atul_laptop.yaml` is the authority. Key values:
`experiment_concurrency: 1`, `cpu_threads: 4`, `dataloader_workers: 0`,
`max_pipeline_ram_gib: 3.5`, `max_cuda_allocated_gib: 4.5`,
`min_cuda_free_before_job_gib: 5.0`, `min_host_free_disk_gib: 40`.

These are **maximum starting budgets**, not assertions about what the
models will consume. Never raise them automatically to make a job
succeed; if a job needs more, say so and get an explicit decision.

* **Do not run training, benchmarking or substantial data generation
  concurrently with another heavy job.** Check `nvidia-smi` and the
  host memory first. A missing process row in `nvidia-smi` is *not*
  proof that the GPU is idle on Windows/WSL.
* The project job lock (`runs/.pdelab.lock`) stops duplicate *PDELab*
  jobs. It is **not** a system-wide GPU scheduler and must never be
  described as one.
* Do not kill another project's process, change its configuration, or
  unload another model.
* No automatic overnight jobs, cloud rentals, large remote downloads,
  driver changes, WSL memory changes or background scheduling.
* Measured usage so far: peak CUDA 0.012-0.023 GiB, peak host RSS
  ~1.5 GiB, 8-23 min per training run. **Wall time is the binding
  constraint, not memory.**

## 3. Result honesty (non-negotiable)

* Never claim an unexecuted study is complete. Report **met / missed /
  not_run / not_comparable** for every declared target.
* A missed target is a valid result. Record it, diagnose a bounded set
  of causes, and preserve the matched comparison.
* Never quietly reduce the difficulty of a condition, shrink a horizon,
  or tune a condition until a target passes. If a configuration is
  changed, change it *harder* when the trivial baseline is the problem,
  and record the reason, the evidence and the date.
* One seed is one seed. Do not claim an uncertainty estimate from
  single-seed runs, and do not present repeated inference of one
  checkpoint as three training seeds.
* Generate every number in the report from a stored artifact. Nothing
  in `docs/RESULTS.md` may be typed by hand.
* Always report the trivial persistence baseline next to any accuracy
  number, and the reference solution's own residual next to any
  residual number.
* Do not backdate implementation, experiments or Git history. Use the
  actual execution dates.
* Do not infer a GitHub identity from a service login. The author is
  Atul Pandey.
* Do not push code, create remote repositories, spend money, or alter
  global provider credentials or permissions.

## 4. Phase gates

Do not skip a gate.

1. `pdelab doctor` must show CUDA available and a successful
   forward/backward.
2. `scripts/linux/validate_solvers.py` must report `OVERALL: PASS`
   (12 checks per system) **before** any data is generated.
3. A solver change requires re-running the full validation suite.
4. `pytest tests/ -q` must pass before a training result is reported.

## 5. Where things are

* Solvers: `src/pdelab/solvers/` (WENO5 Burgers is the primary
  generator; MUSCL and a dealiased spectral scheme are cross-checks;
  the spectral Allen-Cahn solver is the 2D generator).
* Validation suite: `src/pdelab/solvers/validation.py`.
* Physics: `src/pdelab/physics/operators.py` (periodic spectral
  operators) and `losses.py` (the composite objective).
* Models: `src/pdelab/models/fno.py`, `unet.py`.
* Training: `src/pdelab/training/trainer.py` (checkpoint, resume,
  physics-gradient check), `experiment.py` (run manifests).
* Evaluation: `src/pdelab/evaluation/` (rollout, metrics, residual
  evaluators).
* Benchmarks: `src/pdelab/benchmarks/` (process-isolated variants).
* Docs: `docs/`. Read `docs/STATUS.md` and `docs/DECISIONS.md` first.

## 6. Reproduction commands

```bash
cd /home/atulp/projects/pdelab
.venv/bin/python -m pdelab.cli doctor
.venv/bin/python -m pdelab.cli resources check
.venv/bin/python scripts/linux/validate_solvers.py
.venv/bin/python -m pdelab.cli data generate --config configs/pdes/burgers.yaml --profile prototype
.venv/bin/python -m pdelab.cli train --config configs/experiments/burgers_fno1d_data_only_seed0.yaml --profile prototype
.venv/bin/python scripts/linux/evaluate_all.sh
.venv/bin/python scripts/linux/benchmark_all.sh
.venv/bin/python -m pdelab.cli report
.venv/bin/python -m pytest tests/ -q
```

Helper scripts: `scripts/linux/show_run.py <run_id>`,
`scripts/linux/compare_runs.py`, `scripts/linux/summarize_bench.py <run_id>`.

## 7. Known open items

See `docs/NEXT_SESSION.md`. The largest gaps are: only seed 0 has been
trained; the 18-run confirmatory matrix has not been executed; the
physics-regularized U-Net and the λ grid have not been run; and the
Allen-Cahn residual improvement is evaluator-dependent (+20.6% primary,
-2.6% diagnostic) and needs investigating before the confirmatory
matrix.
