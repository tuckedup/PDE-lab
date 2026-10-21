"""Resource preflight and project-level job lock.

The lock prevents *duplicate cooperating PDELab jobs* from running at the
same time. It is **not** a system-wide GPU scheduler and must not be
described as one: it does not know about other projects that may be using
the same GPU.

The preflight reports what it can measure honestly and states explicitly
when a value cannot be established (for example GPU free memory when
another process holds an unaccounted allocation, or the physical free
disk space of the Windows volume that backs the WSL filesystem when the
host cannot be queried).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import threading
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


def windows_physical_disk() -> dict[str, Any]:
    """The physical free disk space of the Windows volume behind WSL.

    The WSL root filesystem is a **virtual disk**: its size (and its
    "free space") is the size of the ext4 image, not the free space of
    the Windows volume that stores that image.  A WSL filesystem can
    report hundreds of gigabytes free while the physical volume is
    nearly full, and only the physical figure says whether the job can
    still write its artifacts.

    The query goes through the WSL interop (``powershell.exe``).  When
    the host cannot be queried the result says so explicitly; the WSL
    figure is **not** substituted for it.
    """
    result: dict[str, Any] = {
        "available": False,
        "source": None,
        "backing_drive": None,
        "free_gib": None,
        "total_gib": None,
        "drives": [],
        "distro_base_path": None,
        "note": None,
    }
    exe = shutil.which("powershell.exe")
    if exe is None:
        result["note"] = (
            "powershell.exe is not reachable from this shell, so the physical "
            "free disk space of the Windows volumes could not be established"
        )
        return result

    distro = (os.environ.get("WSL_DISTRO_NAME") or "").replace("'", "''")
    script = (
        "$lxss='HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss';"
        "$base=$null;"
        "if (Test-Path $lxss) { Get-ChildItem $lxss | ForEach-Object {"
        "$p=Get-ItemProperty $_.PSPath;"
        f"if ($p.DistributionName -eq '{distro}') {{ $base=$p.BasePath }} }} }};"
        "$drives=Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | "
        "Select-Object DeviceID,VolumeName,Size,FreeSpace;"
        "[PSCustomObject]@{distro_base_path=$base; drives=$drives} | "
        "ConvertTo-Json -Depth 4 -Compress"
    )
    try:
        proc = subprocess.run(
            [exe, "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - depends on the host
        result["note"] = f"the Windows disk query failed: {exc}"
        return result
    if proc.returncode != 0:
        result["note"] = (
            "the Windows disk query failed: "
            f"{(proc.stderr or proc.stdout).strip()[:200]}"
        )
        return result
    try:
        payload = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError as exc:
        result["note"] = f"the Windows disk query returned unreadable output: {exc}"
        return result

    base_path = payload.get("distro_base_path")
    result["distro_base_path"] = base_path
    drives = payload.get("drives") or []
    if isinstance(drives, dict):
        drives = [drives]
    for d in drives:
        free = d.get("FreeSpace")
        total = d.get("Size")
        result["drives"].append({
            "device": d.get("DeviceID"),
            "label": d.get("VolumeName"),
            "total_gib": round(total / 1024**3, 3) if total else None,
            "free_gib": round(free / 1024**3, 3) if free else None,
        })

    backing = None
    if isinstance(base_path, str) and len(base_path) > 1 and base_path[1] == ":":
        backing = base_path[:2].upper()
    result["backing_drive"] = backing
    match = next((d for d in result["drives"] if d["device"] == backing), None)
    if match is None and len(result["drives"]) == 1:
        match = result["drives"][0]
        result["backing_drive"] = match["device"]
    if match is None:
        result["note"] = (
            "the Windows volumes were listed but the volume that stores this "
            "WSL filesystem could not be identified"
        )
        return result
    result["available"] = True
    result["source"] = "powershell.exe Win32_LogicalDisk (physical Windows volume)"
    result["free_gib"] = match["free_gib"]
    result["total_gib"] = match["total_gib"]
    return result


def host_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    root = project_root()
    disk = psutil.disk_usage(str(root))
    return {
        "total_ram_gib": round(vm.total / 1024**3, 3),
        "available_ram_gib": round(vm.available / 1024**3, 3),
        "used_ram_percent": vm.percent,
        # The WSL filesystem figure: the size of the ext4 image of the
        # distribution.  It is *not* the physical free space.
        "disk_total_gib": round(disk.total / 1024**3, 3),
        "disk_free_gib": round(disk.free / 1024**3, 3),
        "wsl_filesystem": {
            "path": str(root),
            "total_gib": round(disk.total / 1024**3, 3),
            "free_gib": round(disk.free / 1024**3, 3),
            "scope": (
                "the WSL root filesystem (a virtual disk). Its free space is "
                "NOT the physical free space of the Windows volume that backs "
                "it; the physical figure is reported separately."
            ),
        },
        "windows_physical_disk": windows_physical_disk(),
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

    # ---- disk space: the physical Windows volume and the WSL filesystem ----
    physical = host["windows_physical_disk"]
    if physical.get("available") and physical.get("free_gib") is not None:
        if physical["free_gib"] < profile.min_host_free_disk_gib:
            violations.append(
                f"the physical free disk space of the Windows volume "
                f"{physical.get('backing_drive')} ({physical['free_gib']} GiB) is "
                f"below the required {profile.min_host_free_disk_gib} GiB"
            )
    else:
        # The requirement could not be checked.  That is stated
        # explicitly; the WSL figure is NOT substituted for the
        # physical one.
        warnings.append(
            "the physical free disk space of the Windows volume that backs this "
            f"WSL filesystem could not be established "
            f"({physical.get('note') or 'unknown reason'}); the requirement of "
            f"{profile.min_host_free_disk_gib} GiB was therefore NOT verified. The "
            f"WSL filesystem figure of {host['wsl_filesystem']['free_gib']} GiB is "
            "recorded separately and is NOT the physical free space."
        )
    wsl_free = host["wsl_filesystem"]["free_gib"]
    if wsl_free < profile.min_host_free_disk_gib:
        violations.append(
            f"the WSL filesystem free space {wsl_free} GiB is below the required "
            f"{profile.min_host_free_disk_gib} GiB"
        )

    # ---- free memory: the requirement is not relaxed ----
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
        "disk_scope_note": (
            "the WSL filesystem figure is the size of the distribution's virtual "
            "disk; the physical free space of the Windows volume that backs it "
            "is reported separately under host.windows_physical_disk"
        ),
        "lock_scope_note": (
            "the project job lock excludes cooperating PDELab jobs only. It is "
            "not a system-wide scheduler and it does not protect the GPU "
            "against unrelated software."
        ),
    }


# ----------------------------------------------------------------------
# job lock
# ----------------------------------------------------------------------
#: The locks held by this process, keyed by the resolved lock path.
#:
#: A *project lease* (``JobLock(..., reentrant=True)``) may be joined
#: by a nested plain lock: the nested lock belongs to the same job (a
#: sweep takes the lease once and then locks per run), not to a second
#: one.  Two plain locks in the same process stay mutually exclusive.
_HELD_LOCKS: dict[str, dict[str, Any]] = {}
_HELD_LOCKS_GUARD = threading.Lock()


class JobLock:
    """Cooperative, project-scoped exclusive lock.

    Prevents two PDELab jobs from running concurrently. It says nothing
    about unrelated software using the GPU.

    Parameters
    ----------
    path:
        The lock file.  The default is the project lock
        ``runs/.pdelab.lock``.
    reentrant:
        Take the lock as a *project lease*: a nested lock of the same
        process joins the lease instead of conflicting with it.  The
        lease is then held until the outermost holder releases it.
    """

    def __init__(self, path: str | Path | None = None, *, reentrant: bool = False) -> None:
        if path is None:
            path = project_root() / "runs" / ".pdelab.lock"
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = project_root() / self.path
        self.reentrant = bool(reentrant)
        self._fh = None
        self._joined = False

    def acquire(self, blocking: bool = False) -> bool:
        key = str(self.path)
        with _HELD_LOCKS_GUARD:
            held = _HELD_LOCKS.get(key)
            if held is not None:
                if not held["reentrant"]:
                    # A plain lock of this process already holds it.
                    return False
                held["depth"] += 1
                self._joined = True
                return True

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "w")
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(fh.fileno(), flags)
        except OSError:
            fh.close()
            return False
        fh.write(f"{os.getpid()} {time.time()}\n")
        fh.flush()
        self._fh = fh
        with _HELD_LOCKS_GUARD:
            _HELD_LOCKS[key] = {
                "reentrant": self.reentrant,
                "depth": 1,
                "fh": fh,
            }
        return True

    def release(self) -> None:
        key = str(self.path)
        fh = None
        with _HELD_LOCKS_GUARD:
            held = _HELD_LOCKS.get(key)
            if held is None:
                self._fh = None
                self._joined = False
                return
            held["depth"] -= 1
            if held["depth"] > 0:
                self._joined = False
                return
            _HELD_LOCKS.pop(key, None)
            fh = held["fh"]
        if fh is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()
        self._fh = None
        self._joined = False

    def __enter__(self) -> JobLock:
        if not self.acquire(blocking=False):
            raise RuntimeError(
                f"another PDELab job holds the lock at {self.path}; refusing to run "
                "concurrently (experiment_concurrency=1)"
            )
        return self

    def __exit__(self, *exc) -> None:
        self.release()
