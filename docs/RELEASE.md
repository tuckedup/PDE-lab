# PDELab local release

**Status: a local release artifact of a research prototype.**

This document describes what is in the bundle, how to reproduce it, how
the clean-install verification was performed and - just as importantly -
what the release does **not** claim.

---

## 1. What this release is, and what it is not

This is a **local** release of PDELab, built from one working tree on
one machine (Ubuntu 24.04 under WSL2, one NVIDIA RTX 4060 Laptop GPU).

It is **not**:

* **not a claim of public deployment.** Nothing was published, pushed,
  uploaded or released to any package index. The wheel was never
  uploaded to PyPI and there is no git remote.
* **not a claim of production serving readiness.** There is no service,
  no API, no deployment configuration, no monitoring, no rollback
  procedure and no load test. The bundle is a Python library plus its
  evidence.
* **not a claim of novel state-of-the-art research.** The study compares
  a physics-regularized FNO against a data-only FNO and against U-Nets
  on **two self-defined periodic benchmarks** (1D viscous Burgers, 2D
  scalar Allen-Cahn). It is not a comparison against published work and
  the numbers are not a state-of-the-art claim.

The bundle contains the **library only**. The datasets, the checkpoints,
the run artifacts and the plots are *not* part of it: they stay in the
source tree and are bound to the release by their SHA-256 in
[`RELEASE_MANIFEST.json`](../RELEASE_MANIFEST.json).

---

## 2. What is in the bundle

| item | path |
|---|---|
| the built wheel | `dist/pdelab-0.1.0-py3-none-any.whl` |
| the wheel checksum sidecar | `dist/pdelab-0.1.0-py3-none-any.whl.sha256` |
| the release manifest | `RELEASE_MANIFEST.json` (repository root) |
| the release evidence index | `campaigns/release/EVIDENCE_INDEX.json` |
| the build / verification transcripts | `campaigns/release/` |
| the build and verification scripts | `scripts/linux/release_*` |
| the release test | `tests/integration/test_release.py` |

### The wheel

The wheel is the `pdelab` library and **nothing else**:

* 35 entries, all of them Python source under `pdelab/` plus the wheel
  metadata in `pdelab-0.1.0.dist-info/`;
* it ships **no** datasets (`.h5`, `.npz`, `.npy`), **no** checkpoints
  (`.pt`, `.pth`, `.ckpt`), **no** run artifacts, **no** plots, **no**
  secrets, **no** virtual environment and **no** `campaigns/` scratch
  tree;
* `scripts/linux/release_build.sh` inspects the archive with
  `unzip -l` and **fails the build** if any of those appear;
* `SOURCE_DATE_EPOCH` is pinned to the commit date of `HEAD` and the
  setuptools build tree is removed before every build, so the archive is
  an exact image of `src/`.

The explicit listing is in
[`campaigns/release/wheel_contents.txt`](../campaigns/release/wheel_contents.txt).

### The artifact of this release

| property | value |
|---|---|
| path | `dist/pdelab-0.1.0-py3-none-any.whl` |
| size | 135 259 bytes (451 951 bytes uncompressed) |
| SHA-256 | `0a09217af559d2261e008e05b1c5736b6772b99fedb236e33e7aa709e1ce76a3` |
| entries | 35 |
| built from | commit `c73809950d1f5f902fd5f3b890f5abe0f9505649` (working tree dirty - see the manifest) |
| `SOURCE_DATE_EPOCH` | `1792621523` (2026-10-21T22:25:23Z, the commit date) |

### The reproducibility check (measured, not asserted)

`scripts/linux/release_reproducibility.sh` builds the wheel **three
times in a row** - with the build tree removed before each run - and
compares the digests with the shipped artifact:

```
build 1/3  sha256 0a09217af559d2261e008e05b1c5736b6772b99fedb236e33e7aa709e1ce76a3  (135259 bytes)
build 2/3  sha256 0a09217af559d2261e008e05b1c5736b6772b99fedb236e33e7aa709e1ce76a3  (135259 bytes)
build 3/3  sha256 0a09217af559d2261e008e05b1c5736b6772b99fedb236e33e7aa709e1ce76a3  (135259 bytes)
shipped    sha256 0a09217af559d2261e008e05b1c5736b6772b99fedb236e33e7aa709e1ce76a3
```

Three consecutive builds **and** the shipped wheel have the identical
SHA-256. The record is in
[`campaigns/release/build_reproducibility.json`](../campaigns/release/build_reproducibility.json)
and in the manifest under `artifact.reproducibility_check`.

This is reproducibility **of the working tree that was built**, not of
the commit: the tree was dirty at build time (the manifest lists the
dirty paths), so a checkout of `c738099` alone would not reproduce this
digest.

### The manifest

[`RELEASE_MANIFEST.json`](../RELEASE_MANIFEST.json) records:

* the wheel filename, size, SHA-256 and full entry list;
* the **measured** reproducibility of that wheel (three consecutive
  builds with the pinned `SOURCE_DATE_EPOCH`, and whether the shipped
  artifact matches them);
* the source revision, the branch and the **dirty status** of the tree
  the wheel was built from (the tree *was* dirty - the list of dirty
  paths is in the manifest);
* the Python / PyTorch / numpy versions, both of the project
  environment and of the clean installation;
* the **dataset identities** of the declared cohort (train / val / test
  SHA-256 per system, read from
  `campaigns/evidence-cohort/cohort.yaml`) and the **checkpoint
  identity** of every cohort member (the checkpoint SHA-256 recorded in
  the run's `evaluation.json`, cross-checked against the file on disk);
* the evaluator and protocol identities the artifacts are bound to;
* a pointer to the **evidence index**;
* the **reproduction command**, which was executed against the installed
  CLI, with its real output;
* the record of the clean-environment installation test, including the
  **result of both `pdelab solvers validate` CLI paths** (exit code,
  whether a JSON report was printed, the number of checks and the failed
  checks);
* the **measured limitations** of the study (section 6).

---

## 3. How to reproduce the bundle

From the source tree, with the project's own virtual environment:

```bash
cd /home/atulp/projects/pdelab
scripts/linux/release_all.sh
```

That runs, in order:

```bash
scripts/linux/release_build.sh          # -> dist/*.whl + the content check
scripts/linux/release_reproducibility.sh # -> the three-build SHA-256 check
scripts/linux/release_clean_install.sh  # -> the clean-environment install test
.venv/bin/python scripts/linux/release_manifest.py   # -> RELEASE_MANIFEST.json
```

The only dependency the build adds is `build` itself, and it is
installed into the project `.venv`:

```bash
uv pip install build --python .venv/bin/python
```

No network access is needed: the wheel is built with `--no-isolation`
(the build backend `setuptools` is already in `.venv`) and the clean
environment is populated from the local `uv` cache.

---

## 4. The clean-install verification

The point of this check is to make it **impossible for an accidental
import of the source tree to fake a pass**. Therefore:

* the temporary directory is created **outside** the repository, in
  `/tmp/pdelab_clean_install_XXXXXX`;
* a **fresh** virtual environment is created inside it;
* the built wheel is installed into that environment;
* every fixture runs with the working directory set to the temporary
  directory and with `PDELAB_ROOT` pinned to it;
* the resolved `pdelab.__file__` is printed and checked, and the
  printed `sys.path` contains no entry of the source checkout.

### The exact commands

```bash
# 1. the artifact
cd /home/atulp/projects/pdelab
ls -1 dist/pdelab-*.whl
sha256sum dist/pdelab-0.1.0-py3-none-any.whl

# 2. the temporary environment, OUTSIDE the checkout
CLEAN_DIR=$(mktemp -d /tmp/pdelab_clean_install_XXXXXX)
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 "$CLEAN_DIR/venv"

# 3. install the wheel into it
uv pip install --offline --python "$CLEAN_DIR/venv/bin/python" \
    /home/atulp/projects/pdelab/dist/pdelab-0.1.0-py3-none-any.whl

# 4. CPU only, four threads
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
       NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export PDELAB_ROOT="$CLEAN_DIR"

# 5. the fixtures, all executed from OUTSIDE the checkout
cp /home/atulp/projects/pdelab/campaigns/release/fixtures/one_step_metric_check.py "$CLEAN_DIR/"
cd "$CLEAN_DIR"
"$CLEAN_DIR/venv/bin/python" -c "import pdelab; print(pdelab.__file__)"
"$CLEAN_DIR/venv/bin/pdelab" doctor
"$CLEAN_DIR/venv/bin/pdelab" solvers validate --system burgers
"$CLEAN_DIR/venv/bin/pdelab" solvers validate --system allen_cahn
"$CLEAN_DIR/venv/bin/python" one_step_metric_check.py
```

`scripts/linux/release_clean_install.sh` runs exactly this sequence and
records every command and its real output in
`campaigns/release/clean_install.log` and
`campaigns/release/clean_install.json`.

### What the fixtures are

| fixture | what it proves | exit |
|---|---|---|
| `install_probe` | `pdelab.__file__` resolves inside the temporary environment's `site-packages`, and the source checkout is not on `sys.path` | 0 |
| `cli_doctor` | the installed console script `pdelab` runs and reports the environment | 0 |
| `solvers_validate_cli` | the **CLI** dispatch of the 1D Burgers numerical validation suite (19 checks) | 0 |
| `solvers_validate_cli_allen_cahn` | the **CLI** dispatch of the 2D Allen-Cahn numerical validation suite (14 checks) | 0 |
| `solvers_validate_cli_json_report` | both CLI steps printed a *parseable* JSON report whose `passed` flag is true - not just exit 0 | 0 |
| `solvers_validate_library` | the numerical validation suite itself, called through the installed package | 0 |
| `one_step_metric_check` | the one-step nRMSE path (`teacher_forced_one_step` -> `nrmse_per_trajectory` -> `summarize_nrmse`) and the two model-free baselines | 0 |
| `reproduction_command` | the exact command recorded in the manifest, executed verbatim against the installed CLI | 0 |

All eight fixtures exited 0 in the recorded run
(`campaigns/release/clean_install.json`).

---

## 5. The reproduction command

The manifest records this command (it lives in
[`campaigns/release/fixtures/reproduction_command.txt`](../campaigns/release/fixtures/reproduction_command.txt),
which is the single source of truth: the clean-install verification
executes that file verbatim and the manifest records the same bytes):

```bash
pdelab doctor && pdelab solvers validate --system burgers
```

It was executed **against the installed wheel**, in the clean
environment, from outside the source checkout, with the release virtual
environment on `PATH`. The recorded output and the exit code are in
`campaigns/release/clean_install/reproduction_command.out` and in the
manifest itself. The run ends with the JSON report of the Burgers
validation suite:

```json
{
  ...
  "passed": true,
  "walltime_seconds": 22.743
}
```

The first part is the installed CLI's own environment check; the second
part re-runs the study's declared numerical gate **through the installed
CLI**.

### Resolved defect: `pdelab solvers validate` now works

The previous revision of this document recorded an open blocker: the
CLI dispatch of the numerical validation suite was broken.

```
ImportError: cannot import name 'run_validation' from 'pdelab.solvers.validation'
```

Root cause: commit `c738099` ("PDELab v2") removed the dispatch
function `run_validation()` from `src/pdelab/solvers/validation.py`,
while `src/pdelab/cli.py:199` still imported it. The function existed as
of commit `2d3ae32`.

**This is fixed in the source that this release was built from.** A
`run_validation(system, verbose=False)` dispatcher exists again in
`src/pdelab/solvers/validation.py`, and **both** CLI paths now exit 0
and print a JSON report from the *installed wheel*:

| command | exit | report |
|---|---|---|
| `pdelab solvers validate --system burgers` | 0 | 19 checks, `passed: true`, no failed check |
| `pdelab solvers validate --system allen_cahn` | 0 | 14 checks, `passed: true`, no failed check |

The clean-install verification exercises both of them - and parses the
two JSON reports, so that "it exited 0 but printed nothing useful"
cannot pass. The manifest records the result under
`clean_install.solver_validation_cli`, and
`clean_install.open_blockers` is empty for this release.

---

## 6. Known limitations of the study, as measured

These are the measured limitations of the *study*, not of the packaging.
Every number below is taken from a stored artifact; none is typed by
hand. The full per-run tables are in the manifest under
`measured_limitations`.

### 6.1 The learned operators do not beat direct numerical stepping

On both benchmarks the **parameter-informed numerical baseline** - a
direct spectral stepping of the PDE itself on the same grid and horizon
- has a *lower* held-out rollout nRMSE than **every** learned operator
of the cohort:

| system | learned rollout nRMSE (cohort range) | numerical stepping nRMSE |
|---|---|---|
| 1D Burgers | 0.00987 - 0.02456 | 0.000536 |
| 2D Allen-Cahn | 0.01285 - 0.05139 | 0.002746 |

That is a factor of between about **4.7x and 46x** in favour of direct
numerical integration (smallest factor: the Allen-Cahn data-only FNO at
seed 2, 0.012848 against 0.002746; largest factor: the Burgers
physics-regularized FNO at seed 1, 0.024561 against 0.000536). The
numerical baseline is *not* parameter free - it needs the PDE and its
parameters - so it is not a like-for-like comparison with a learned
model; it quantifies how much of the task is solvable by direct
numerical integration at negligible cost. The honest reading is that
**the operators do not pay for themselves on these benchmarks**.

### 6.2 The Burgers residual improvement is evaluator-specific

The physics-regularized FNO improves on the data-only FNO under the
**declared** implicit-Euler (first order) residual - about **+70 % per
seed** - and is **worse** under the **independently declared** centred
first-derivative residual, by **45 % to 63 %**:

| system | seed | implicit-Euler (primary) | centred first derivative (independent) |
|---|---|---|---|
| Burgers | 0 | +69.53 % | **-45.41 %** |
| Burgers | 1 | +69.20 % | **-59.94 %** |
| Burgers | 2 | +70.97 % | **-63.37 %** |
| Allen-Cahn | 0 | +21.27 % | +19.46 % |
| Allen-Cahn | 1 | +24.25 % | +22.48 % |
| Allen-Cahn | 2 | +22.35 % | +20.63 % |

The absolute residual RMS values make the second half of the statement
concrete. Under the independent centred first-derivative evaluator the
**reference solution is more PDE-consistent than either learned model**
(Burgers, all seeds: reference 0.0413 vs data-only FNO 0.1011-0.1120
and physics-regularized FNO 0.1629-0.1667). Under the same evaluator
the physics-regularized FNO is *further* from the reference than the
data-only FNO it was supposed to improve on.

The two evaluators are never mixed and are evaluated on the same
support (the interior snapshot indices `1 .. T-2`). The Burgers
improvement is therefore a property of the **declared residual
operator**, not evidence that the model is physically better in general:
a residual penalty can overfit one discretisation of the operator.
Allen-Cahn, in contrast, improves under **both** evaluators.

### 6.3 The inference memory claim is about peak *allocated* memory only

The inference result is a **peak allocated** memory measurement. It is
**not** a total device-memory saving. From the stored benchmark artifact
of the declared inference reference run
(`runs/burgers_fno1d_data_only_seed0/benchmark_matched.json`):

| quantity | FP32 eager | `compile_reduce_overhead` | change |
|---|---|---|---|
| peak **allocated** | 0.008346 GiB | 0.000329 GiB | **-96.05 %** |
| peak **reserved** | 0.021484 GiB | 0.023438 GiB | **+9.09 %** |
| device free memory | 6.879883 GiB | 6.866211 GiB | -0.013672 GiB (-0.20 %) |
| persistent weights + input + output | 196 548 B | 196 548 B | unchanged |

So: the peak *allocated* memory falls by about 96 %, but the peak
*reserved* memory **rises** by about 9 % - the CUDA-graph private pool
is larger than the eager allocator's working set - and the device-level
free-memory reading is essentially unchanged. The claim is about the
allocator's working set, not about the device.

Only one variant passes the joint gate (accuracy within tolerance **and**
throughput **and** peak-allocated memory): `compile_reduce_overhead`.
`autocast_fp16` and `compile_default` fail it, and the per-variant
verdict is recorded in the artifact's `per_variant_joint_pass` list and
in the manifest under
`measured_limitations.inference_memory_claim_is_peak_allocated_only`.

### 6.4 Other limits of the evidence

* One machine, one GPU, three training seeds per condition.
* Two self-defined periodic benchmarks; no comparison against published
  work.
* GPU training is not bitwise deterministic even with the determinism
  settings, so the seeds carry real spread.
* The release was built from a **dirty** working tree; the manifest
  records the exact dirty paths. The wheel is byte-reproducible from
  *that* tree (three consecutive builds, one SHA-256), not from the
  commit alone.
* This is a **local** release artifact: not a public deployment, not
  production serving readiness, and not a novel state-of-the-art claim.

---

## 7. The release test

```bash
cd /home/atulp/projects/pdelab
.venv/bin/python -m pytest tests/integration/test_release.py -q
```

The test asserts that the manifest exists, is valid JSON, that the
recorded SHA-256 matches the wheel on disk and that the recorded
reproduction command is non-empty. It is fast (well under a second) and
it **never rebuilds the wheel**: when the wheel is absent the checksum
comparison is skipped with a clear reason instead of failing.

---

## 8. Where the evidence lives

| what | where |
|---|---|
| the wheel and its checksum | `dist/` |
| the release manifest | `RELEASE_MANIFEST.json` |
| the evidence index (with SHA-256 per artifact) | `campaigns/release/EVIDENCE_INDEX.json` |
| the declared cohort | `campaigns/evidence-cohort/cohort.yaml` |
| the generated result tables | `results/RESULTS.md`, `docs/RESULTS.md` |
| the build transcript | `campaigns/release/build.log` |
| the reproducibility record | `campaigns/release/build_reproducibility.json`, `campaigns/release/build_reproducibility.log` |
| the clean-install transcript | `campaigns/release/clean_install.log` |
| the per-step outputs | `campaigns/release/clean_install/*.out` |

---

## 9. Reproducing the *results* (not the bundle)

The bundle is the library; the results live in the source tree and are
reproduced by the commands in [`docs/REPRODUCE.md`](REPRODUCE.md). The
phase gates still apply: `pdelab doctor` must report CUDA and a
successful forward/backward, the numerical validation suite must report
`OVERALL: PASS` before any data is generated, and `pytest tests/ -q`
must pass before a training result is reported.
