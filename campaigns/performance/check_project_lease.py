"""Check the project lease of the heavy entrypoints.

The lease is the project job lock (``runs/.pdelab.lock``).  It excludes
**cooperating PDELab jobs only**: it is not a system-wide scheduler and
it does not protect the machine against unrelated software that may be
using the same GPU.

The check proves three things:

1. a heavy entrypoint refuses to start while another PDELab job holds
   the lease (exit code 3), and starts normally when the lease is free;
2. two plain locks in the same process stay mutually exclusive;
3. a nested plain lock *joins* the project lease of the same job (a
   sweep takes the lease once and then locks per run).

Usage::

    .venv/bin/python campaigns/performance/check_project_lease.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from pdelab.config import project_root
from pdelab.resources.preflight import JobLock

LOCK_PATH = project_root() / "runs" / ".pdelab.lock"


def _hold_lock(seconds: float) -> subprocess.Popen:
    """A background process that holds the project lease."""
    script = (
        "import time, sys;"
        "from pdelab.resources.preflight import JobLock;"
        f"lock = JobLock({str(LOCK_PATH)!r});"
        "assert lock.acquire(blocking=False);"
        f"time.sleep({seconds});"
    )
    return subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(project_root()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_benchmark() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pdelab.cli", "benchmark", "--run-id", "lease_probe"],
        cwd=str(project_root()),
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.returncode, (proc.stderr or "").strip()


def _other_process_can_lock(path: str) -> bool:
    """Whether a *different* process can take the lock at ``path``."""
    script = (
        "import sys;"
        "from pdelab.resources.preflight import JobLock;"
        f"lock = JobLock({path!r});"
        "sys.exit(0 if lock.acquire(blocking=False) else 1)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(project_root()),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode == 0


def _run_cli(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pdelab.cli", *args],
        cwd=str(project_root()),
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc.returncode, (proc.stderr or "").strip()


#: Every heavy entrypoint, with the arguments it needs to get as far as
#: the lease.  None of them may start any work while the lease is held.
HEAVY_ENTRYPOINTS = [
    ("train", [
        "train", "--config",
        "configs/experiments/burgers_fno1d_data_only_seed0.yaml",
        "--profile", "smoke",
    ]),
    ("evaluate", [
        "evaluate", "--run-id", "lease_probe", "--device", "cpu", "--split", "val",
    ]),
    ("benchmark", ["benchmark", "--run-id", "lease_probe", "--device", "cuda"]),
    ("sweep", ["sweep", "--config", "/tmp/pdelab_perf/sweep_matrix.yaml"]),
    ("data generate", [
        "data", "generate", "--config", "configs/pdes/burgers.yaml",
        "--profile", "smoke", "--out", "/tmp/pdelab_perf/lease_probe_data",
    ]),
]

SWEEP_MATRIX = """
matrix:
  systems: [burgers]
  architectures: [fno1d]
  conditions: [data_only]
  seeds: [0]
"""


def main() -> int:
    failures = []
    matrix_path = Path("/tmp/pdelab_perf/sweep_matrix.yaml")
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(SWEEP_MATRIX, encoding="utf-8")

    # 1. the lease is free -> the entrypoint gets past the lease
    code, err = _run_benchmark()
    print(f"lease free      -> exit code {code}")
    if code == 3:
        failures.append("the entrypoint was refused although the lease was free")
    if "project lock" in err:
        failures.append("the entrypoint reported a lock conflict although the lease was free")

    # 2. another PDELab job holds the lease -> every heavy entrypoint refuses
    holder = _hold_lock(40.0)
    try:
        time.sleep(2.0)
        for name, argv in HEAVY_ENTRYPOINTS:
            code, err = _run_cli(argv)
            refused = code == 3 and "project lock" in err
            print(f"lease held -> {name:14s} exit code {code} refused={refused}")
            if not refused:
                failures.append(f"'{name}' was not refused while the lease was held")
            if "cooperating PDELab jobs only" not in err:
                failures.append(
                    f"the refusal of '{name}' does not say that the lock excludes "
                    "cooperating PDELab jobs only"
                )
    finally:
        holder.terminate()
        holder.wait(timeout=30)

    # 3. two plain locks in the same process are mutually exclusive
    a = JobLock(Path("/tmp/pdelab_lease_probe_lock"))
    b = JobLock(Path("/tmp/pdelab_lease_probe_lock"))
    if not a.acquire(blocking=False):
        failures.append("the first plain lock could not be taken")
    if b.acquire(blocking=False):
        failures.append("a second plain lock was granted: the lock is not exclusive")
        b.release()
    a.release()
    print("two plain locks -> mutually exclusive")

    # 4. a nested plain lock joins the project lease of the same job
    probe_path = "/tmp/pdelab_lease_probe_lock"
    lease = JobLock(Path(probe_path), reentrant=True)
    nested = JobLock(Path(probe_path))
    lease.acquire(blocking=False)
    joined = nested.acquire(blocking=False)
    print(f"nested lock     -> joined the lease: {joined}")
    if not joined:
        failures.append("a nested plain lock did not join the project lease")

    # the lease is held against *other* processes for as long as it lives
    if _other_process_can_lock(probe_path):
        failures.append("another process could take the lease while it was held")
    nested.release()
    if _other_process_can_lock(probe_path):
        failures.append("the nested release dropped the project lease")
    else:
        print("after the nested release the lease is still held")
    lease.release()
    if not _other_process_can_lock(probe_path):
        failures.append("the lease was not released")
    else:
        print("after the lease release the lock is free again")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("all lease checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
