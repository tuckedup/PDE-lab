"""Resource preflight and project-level job lock.

The lock prevents *duplicate cooperating PDELab jobs* from running at the
same time. It is **not** a system-wide GPU scheduler and must not be
described as one: it does not know about other projects that may be using
the same GPU.

The preflight reports what it can measure honestly and states explicitly
when a value cannot be established (for example GPU free memory when
another process holds an unaccounted allocation).
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from ..config import project_root


@dataclass
class HardwareProfile:
    """Conservative engineering limits for the development machine."""

    experiment_concurrency: int = 1
    cpu_threads: int = 4
    dataloader_workers: int = 0
    max_pipeline_ram_gib: float = 3.5
    max_cuda_allocated_gib: float = 4.5
    min_cuda_free_before_job_gib: float = 5.0
    min_host_free_disk_gib: float = 40.0
    max_new_dataset_storage_gib: float = 4.0
    max_new_project_storage_gib: float = 10.0
    max_checkpoints_per_run: int = 2
    initial_batch_size_1d: int = 8
    initial_batch_size_2d: int = 2
    initial_trial_walltime_minutes: float = 20.0
    max_default_job_walltime_minutes: float = 45.0
    max_default_sweep_walltime_minutes: float = 120.0
    training_seeds_prototype: list[int] = None  # type: ignore[assignment]
    training_seeds_confirmatory: list[int] = None  # type: ignore[assignment]
    run_full_sweep_by_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def load(path: str | Path) -> HardwareProfile:
        import yaml

        p = Path(path)
        if not p.is_absolute():
            p = project_root() / p
        with open(p, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        known = {f for f in HardwareProfile.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        profile = HardwareProfile(**filtered)
        if profile.training_seeds_prototype is None:
            profile.training_seeds_prototype = [0]
        if profile.training_seeds_confirmatory is None:
            profile.training_seeds_confirmatory = [0, 1, 2]
        return profile


def _gpu_info() -> dict[str, Any]:
    """Query GPU memory/utilisation. Reports honestly when unavailable."""
    info: dict[str, Any] = {
        "available": False,
        "name": None,
        "total_gib": None,
        "free_gib": None,
        "used_gib": None,
        "utilization_percent": None,
        "processes": [],
        "note": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            info["note"] = "torch reports no CUDA device"
            return info
        free, total = torch.cuda.mem_get_info(0)
        info["available"] = True
        info["name"] = torch.cuda.get_device_name(0)
        info["total_gib"] = round(total / 1024**3, 3)
        info["free_gib"] = round(free / 1024**3, 3)
        info["used_gib"] = round((total - free) / 1024**3, 3)
        try:
            info["utilization_percent"] = torch.cuda.utilization(0)
        except Exception:  # pragma: no cover - depends on driver
            info["utilization_percent"] = None
    except Exception as exc:  # pragma: no cover - defensive
        info["note"] = f"GPU query failed: {exc}"
    return info


def _nvidia_smi() -> dict[str, Any]:
    """Attempt an nvidia-smi query for process attribution."""
    import shutil
    import subprocess

    result: dict[str, Any] = {"available": False, "processes": [], "note": None}
    exe = shutil.which("nvidia-smi")
    if exe is None:
        result["note"] = "nvidia-smi not on PATH"
        return result
    try:
        proc = subprocess.run(
            [exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if proc.returncode == 0:
            result["available"] = True
            for line in proc.stdout.strip().splitlines():
                if not line.strip():
                    continue
                pid, mem = [p.strip() for p in line.split(",")]
                result["processes"].append({"pid": pid, "used_memory": mem})
        else:
            result["note"] = proc.stderr.strip()[:200]
    except Exception as exc:  # pragma: no cover
        result["note"] = f"nvidia-smi failed: {exc}"
    return result


def host_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(project_root()))
    return {
        "total_ram_gib": round(vm.total / 1024**3, 3),
        "available_ram_gib": round(vm.available / 1024**3, 3),
        "used_ram_percent": vm.percent,
        "disk_total_gib": round(disk.total / 1024**3, 3),
        "disk_free_gib": round(disk.free / 1024**3, 3),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "rss_gib": round(psutil.Process(os.getpid()).memory_info().rss / 1024**3, 4),
    }


def preflight(profile: HardwareProfile, require_gpu: bool = True) -> dict[str, Any]:
    """Run the preflight checks and return a structured report.

    The report contains a list of violations. An empty list means the
    job is within the declared budget.
    """
    gpu = _gpu_info()
    smi = _nvidia_smi()
    host = host_info()
    violations: list[str] = []
    warnings: list[str] = []

    if require_gpu and not gpu["available"]:
        violations.append("no CUDA device available but a GPU job was requested")
    if gpu["free_gib"] is not None and gpu["free_gib"] < profile.min_cuda_free_before_job_gib:
        violations.append(
            f"GPU free memory {gpu['free_gib']} GiB is below the required "
            f"{profile.min_cuda_free_before_job_gib} GiB"
        )
    if host["disk_free_gib"] < profile.min_host_free_disk_gib:
        violations.append(
            f"host free disk {host['disk_free_gib']} GiB is below the required "
            f"{profile.min_host_free_disk_gib} GiB"
        )
    if host["available_ram_gib"] < profile.max_pipeline_ram_gib:
        warnings.append(
            f"available RAM {host['available_ram_gib']} GiB is close to the pipeline "
            f"budget {profile.max_pipeline_ram_gib} GiB"
        )
    if smi["processes"]:
        warnings.append(
            f"{len(smi['processes'])} other compute process(es) reported on the GPU"
        )
    if gpu["available"] and not smi["available"]:
        warnings.append(
            "nvidia-smi process attribution unavailable; GPU occupancy cannot be "
            "independently confirmed"
        )

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host,
        "gpu": gpu,
        "nvidia_smi": smi,
        "profile": profile.to_dict(),
        "violations": violations,
        "warnings": warnings,
        "ok": not violations,
    }


# ----------------------------------------------------------------------
# job lock
# ----------------------------------------------------------------------
class JobLock:
    """Cooperative, project-scoped exclusive lock.

    Prevents two PDELab jobs from running concurrently. It says nothing
    about unrelated software using the GPU.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = project_root() / "runs" / ".pdelab.lock"
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = project_root() / self.path
        self._fh = None

    def acquire(self, blocking: bool = False) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(self._fh.fileno(), flags)
        except OSError:
            self._fh.close()
            self._fh = None
            return False
        self._fh.write(f"{os.getpid()} {time.time()}\n")
        self._fh.flush()
        return True

    def release(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None

    def __enter__(self) -> JobLock:
        if not self.acquire(blocking=False):
            raise RuntimeError(
                f"another PDELab job holds the lock at {self.path}; refusing to run "
                "concurrently (experiment_concurrency=1)"
            )
        return self

    def __exit__(self, *exc) -> None:
        self.release()
