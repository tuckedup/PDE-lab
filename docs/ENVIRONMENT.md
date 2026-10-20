# PDELab - Environment

## Execution boundary

PDELab runs **inside Ubuntu WSL2**; the agent (OpenCode) runs on Windows.

| | |
|---|---|
| Windows host | MSI laptop, Windows 11, AMD Ryzen 9 8945HS, 16 logical CPUs |
| Linux runtime | Ubuntu 24.04.1 LTS on WSL2, user `atulp` |
| Canonical project directory | `/home/atulp/projects/pdelab` |
| Windows view of the same directory | `\\wsl.localhost\Ubuntu\home\atulp\projects\pdelab` |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8,188 MiB VRAM, driver 581.80 |
| WSL-visible RAM | ~7.4 GiB (host ~15.3 GiB) |
| Python | 3.11.17 in an isolated `.venv` created by `uv` |
| PyTorch | 2.14.1+cu130 (CUDA 13.0) |

There is **one** repository and **one** Linux dependency environment.
Nothing is duplicated into a second Windows directory.

## Environment creation (from scratch)

```bash
# Windows PowerShell
wsl -d Ubuntu --cd /home/atulp -- mkdir -p projects/pdelab
wsl -d Ubuntu --cd /home/atulp/projects/pdelab -- pwd
```

```bash
# inside WSL
cd /home/atulp/projects/pdelab
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv is not preinstalled
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
```

`uv` installs to `~/.local/bin`, which is **not** on the default PATH of a
fresh WSL shell. Either `export PATH="$HOME/.local/bin:$PATH"` or call
`/home/atulp/.local/bin/uv` by absolute path.

Do **not** install a Linux NVIDIA display driver in WSL, and do not
modify the Windows driver.

## Verification

```bash
.venv/bin/python -m pdelab.cli doctor
```

Expected on this machine: CUDA `available: true`, device
`NVIDIA GeForce RTX 4060 Laptop GPU`, `forward_backward_ok: true`.

## Resource profile

`configs/hardware/atul_laptop.yaml` holds the conservative engineering
limits. The values are **maximum starting budgets**, not assertions about
what the models consume, and they are never raised automatically to
overcome a failure.

| limit | value |
|---|---|
| experiment_concurrency | 1 |
| cpu_threads | 4 |
| dataloader_workers | 0 |
| max_pipeline_ram_gib | 3.5 |
| max_cuda_allocated_gib | 4.5 |
| min_cuda_free_before_job_gib | 5.0 |
| min_host_free_disk_gib | 40 |
| max_new_dataset_storage_gib | 4 |
| max_new_project_storage_gib | 10 |
| max_checkpoints_per_run | 2 |
| initial_batch_size_1d | 8 |
| initial_batch_size_2d | 2 |
| max_default_job_walltime_minutes | 45 |
| max_default_sweep_walltime_minutes | 120 |
| training_seeds_prototype | [0] |
| training_seeds_confirmatory | [0, 1, 2] |
| run_full_sweep_by_default | false |

`min_cuda_free_before_job_gib` was lowered from the 6.0 in the project
brief to **5.0** after the first measurement: the RTX 4060 reported
5.8-6.9 GiB free while idle (Windows holds ~1-2 GiB for graphics), so a
6.0 GiB gate would have blocked every GPU job on a quiet machine.

## Concurrency and the job lock

`pdelab.resources.preflight.JobLock` takes an exclusive `flock` on
`runs/.pdelab.lock` so that two *cooperating PDELab jobs* cannot run at
the same time.

The lock is **not** a system-wide GPU scheduler. It does not know about
other projects (for example FlyCircuit) that may be using the same GPU,
and it must never be described as preventing that. Before a job starts,
`pdelab resources check` reports:

* host RAM and free disk,
* GPU free / used / total memory,
* the `nvidia-smi` compute-process list.

On Windows/WSL, device process attribution can be incomplete. A missing
process row is **not** proof that the GPU is idle; the report says so
explicitly when `nvidia-smi` attribution is unavailable.

## Measured resource use of the completed prototype runs

| run | peak CUDA allocated | peak host RSS | train wall time |
|---|---|---|---|
| burgers_fno1d_data_only_seed0 | 0.0186 GiB | 1.41 GiB | 635 s |
| burgers_fno1d_physics_seed0 | 0.0193 GiB | 1.53 GiB | 800 s |
| burgers_unet1d_data_only_seed0 | 0.0196 GiB | 1.45 GiB | 1289 s |
| allen_cahn_fno2d_data_only_seed0 | 0.0215 GiB | 1.45 GiB | 516 s |
| allen_cahn_fno2d_physics_seed0 | 0.0225 GiB | 1.56 GiB | 642 s |
| allen_cahn_unet2d_data_only_seed0 | 0.0118 GiB | 1.52 GiB | 1372 s |

Every run used well under 1% of the 8 GiB VRAM and about 1.5 GiB of the
7.4 GiB visible to WSL. The limiting resource in this study was
**wall time**, not memory.

Dataset storage: `data/burgers` 4.1 MiB, `data/allen_cahn` 7.2 MiB
(compressed HDF5), against a 4 GiB budget.

## Git

The repository is local only: there is no remote, and nothing is pushed.
The commit revision and dirty status are recorded in every run manifest
so a result can be tied to a code state.
