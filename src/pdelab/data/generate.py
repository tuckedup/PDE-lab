"""Dataset generation, storage and manifests.

Storage format
--------------
**HDF5** (chunked, float32) with one file per (system, split) plus a
JSON sidecar manifest. Trajectories are written incrementally, one
trajectory at a time, so the process never holds the full dataset in
memory. Files are compressed with gzip to bound disk usage.

Restriction from the reference grid to the observation grid
-----------------------------------------------------------
The reference solution is computed on a *finer* grid than the
observation grid. The restriction is **not** a naive stride. Instead the
reference field is truncated in Fourier space to the observation
band-limit (removing the modes that would alias onto the observation
grid) and only then sampled at the observation points. This is recorded
as ``restriction: spectral_truncate_then_subsample`` in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..config import PDESystemConfig, project_root
from .initial_conditions import fourier_series_1d, fourier_series_2d


# ----------------------------------------------------------------------
# restriction
# ----------------------------------------------------------------------
def restrict_1d(fine: np.ndarray, n_obs: int) -> np.ndarray:
    """Restrict a fine periodic 1D field to ``n_obs`` points.

    Modes above the observation band-limit are removed *before* sampling
    so that no aliased content reaches the observation grid. A symmetric
    ``|k|`` mask on the full FFT is used so that the positive and
    negative modes are treated identically.
    """
    n_fine = fine.shape[-1]
    if n_fine == n_obs:
        return fine.copy()
    if n_fine % n_obs != 0:
        raise ValueError(
            f"fine grid {n_fine} is not an integer multiple of observation grid {n_obs}"
        )
    stride = n_fine // n_obs
    k = np.fft.fftfreq(n_fine) * n_fine
    mask = np.abs(k) <= n_obs // 2
    spectrum = np.fft.fft(fine) * mask
    filtered = np.fft.ifft(spectrum).real
    return filtered[::stride].copy()


def restrict_2d(fine: np.ndarray, nx_obs: int, ny_obs: int) -> np.ndarray:
    """Restrict a fine periodic 2D field to the observation grid.

    Modes above the observation band-limit are removed *before* sampling
    so that no aliased content reaches the observation grid.

    The full complex FFT is used with a symmetric ``|k|`` mask. Using
    ``rfft2`` here would be incorrect: it retains *both* signs of the
    first (non-halved) axis, so a one-sided row truncation would delete
    the negative-kx modes and silently halve the field.
    """
    nx_fine, ny_fine = fine.shape[-2:]
    if (nx_fine, ny_fine) == (nx_obs, ny_obs):
        return fine.copy()
    if nx_fine % nx_obs != 0 or ny_fine % ny_obs != 0:
        raise ValueError(
            f"fine grid {(nx_fine, ny_fine)} is not an integer multiple of "
            f"observation grid {(nx_obs, ny_obs)}"
        )
    sx = nx_fine // nx_obs
    sy = ny_fine // ny_obs

    kx = np.fft.fftfreq(nx_fine) * nx_fine
    ky = np.fft.fftfreq(ny_fine) * ny_fine
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    mask = (np.abs(KX) <= nx_obs // 2) & (np.abs(KY) <= ny_obs // 2)

    spectrum = np.fft.fft2(fine) * mask
    filtered = np.fft.ifft2(spectrum).real
    return filtered[::sx, ::sy].copy()


# ----------------------------------------------------------------------
# manifests
# ----------------------------------------------------------------------
def git_revision(root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"revision": None, "dirty": None, "branch": None}
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10, check=False,
        )
        if rev.returncode == 0:
            info["revision"] = rev.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root, capture_output=True, text=True, timeout=10, check=False,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10, check=False,
        )
        if branch.returncode == 0:
            info["branch"] = branch.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def environment_snapshot() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


@dataclass
class TrajectoryRecord:
    """Per-trajectory provenance recorded in the manifest."""

    index: int
    split: str
    ic_seed: int
    params: dict[str, float]
    grid: tuple[int, ...]
    observation_times: list[float]
    solver: str
    solver_version: str
    ic_spectrum: dict[str, Any]
    n_ref_steps: int
    ref_dt: float
    rollout_hash: str


# ----------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------
class DatasetGenerator:
    """Generate, store and describe a benchmark dataset."""

    def __init__(
        self,
        system: PDESystemConfig,
        out_dir: str | Path,
        n_train: int,
        n_val: int,
        n_test: int,
        seed: int,
        profile: str = "prototype",
    ) -> None:
        self.system = system
        self.out_dir = Path(out_dir)
        if not self.out_dir.is_absolute():
            self.out_dir = project_root() / self.out_dir
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.seed = int(seed)
        self.profile = profile
        self.records: list[TrajectoryRecord] = []

    # ------------------------------------------------------------------
    def _make_solver(self):
        from ..solvers.allen_cahn import AllenCahn2DSpectral
        from ..solvers.burgers_weno import Burgers1DWeno5

        if self.system.dimension == 1:
            return Burgers1DWeno5(
                n=self.system.reference_grid[0],
                nu=self.system.params.get("nu", 0.01),
            )
        return AllenCahn2DSpectral(
            nx=self.system.reference_grid[0],
            ny=self.system.reference_grid[1],
            D=self.system.params.get("D", 0.001),
            a=self.system.params.get("a", 1.0),
        )

    def _make_ic(self, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        ic_cfg = dict(self.system.initial_condition)
        kind = ic_cfg.pop("kind", "fourier")
        if self.system.dimension == 1:
            return fourier_series_1d(
                self.system.reference_grid[0], rng, **ic_cfg
            )
        if kind == "gaussian":
            from .initial_conditions import gaussian_field_2d

            return gaussian_field_2d(
                self.system.reference_grid[0], self.system.reference_grid[1], rng, **ic_cfg
            )
        return fourier_series_2d(
            self.system.reference_grid[0], self.system.reference_grid[1], rng, **ic_cfg
        )

    def _trajectory_shape(self) -> tuple[int, ...]:
        n_times = len(self.system.observation_times)
        return (n_times, *self.system.observation_grid)

    # ------------------------------------------------------------------
    def generate(self, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Generate all splits and write the dataset + manifest."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        solver = self._make_solver()
        obs_times = np.asarray(self.system.observation_times, dtype=np.float64)
        shape = self._trajectory_shape()
        n_total = self.n_train + self.n_val + self.n_test

        split_names = (
            ["train"] * self.n_train
            + ["val"] * self.n_val
            + ["test"] * self.n_test
        )

        # Deterministic per-trajectory seeds derived from the master seed.
        seed_sequence = np.random.SeedSequence(self.seed)
        child_seeds = seed_sequence.spawn(n_total)

        files: dict[str, Path] = {}
        handles: dict[str, h5py.File] = {}
        try:
            for split in ["train", "val", "test"]:
                path = self.out_dir / f"{self.system.name}_{split}.h5"
                files[split] = path
                handle = h5py.File(path, "w")
                handles[split] = handle
                handle.create_dataset(
                    "u",
                    shape=(split_names.count(split), *shape),
                    dtype="float32",
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, *shape),
                )
                handle.create_dataset(
                    "ic_seeds", shape=(split_names.count(split),), dtype="int64"
                )

            counters = {"train": 0, "val": 0, "test": 0}
            t_start = time.time()
            for idx in range(n_total):
                split = split_names[idx]
                ic_seed = int(child_seeds[idx].generate_state(1, dtype=np.uint32)[0])
                rng = np.random.default_rng(ic_seed)
                u0_ref, spec = self._make_ic(rng)

                ref = solver.solve_to(u0_ref, self.system.t_final, obs_times)
                if not np.all(np.isfinite(ref)):
                    raise FloatingPointError(
                        f"non-finite reference trajectory at index {idx} (seed {ic_seed})"
                    )

                if self.system.dimension == 1:
                    obs = np.stack(
                        [restrict_1d(frame, self.system.observation_grid[0]) for frame in ref]
                    )
                else:
                    obs = np.stack(
                        [
                            restrict_2d(
                                frame,
                                self.system.observation_grid[0],
                                self.system.observation_grid[1],
                            )
                            for frame in ref
                        ]
                    )

                slot = counters[split]
                handles[split]["u"][slot] = obs.astype(np.float32)
                handles[split]["ic_seeds"][slot] = ic_seed
                counters[split] += 1

                rollout_bytes = obs.astype(np.float32).tobytes()
                self.records.append(
                    TrajectoryRecord(
                        index=idx,
                        split=split,
                        ic_seed=ic_seed,
                        params=dict(self.system.params),
                        grid=tuple(self.system.observation_grid),
                        observation_times=[float(t) for t in obs_times],
                        solver=self.system.solver,
                        solver_version=solver_version(),
                        ic_spectrum=spec,
                        n_ref_steps=int(getattr(solver, "_last_steps", 0) or 0),
                        ref_dt=float(getattr(solver, "_last_dt", 0.0) or 0.0),
                        rollout_hash=hashlib.sha256(rollout_bytes).hexdigest(),
                    )
                )
                if progress and (idx + 1) % 10 == 0:
                    progress(f"  generated {idx + 1}/{n_total} trajectories")
            elapsed = time.time() - t_start
        finally:
            for handle in handles.values():
                handle.close()

        manifest = self._write_manifest(files, elapsed)
        return manifest

    # ------------------------------------------------------------------
    def _write_manifest(self, files: dict[str, Path], elapsed: float) -> dict[str, Any]:
        root = project_root()
        manifest = {
            "dataset": self.system.name,
            "profile": self.profile,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generation_walltime_seconds": round(elapsed, 3),
            "system": asdict(self.system),
            "splits": {
                "train": self.n_train,
                "val": self.n_val,
                "test": self.n_test,
            },
            "split_seed": self.seed,
            "split_policy": "trajectory_level_by_initial_condition_seed",
            "files": {
                split: {
                    "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                    "sha256": _file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for split, path in files.items()
            },
            "trajectories": [asdict(r) for r in self.records],
            "git": git_revision(root),
            "environment": environment_snapshot(),
        }
        manifest_path = self.out_dir / f"{self.system.name}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        manifest["manifest_path"] = str(manifest_path)
        return manifest


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def solver_version() -> str:
    from .. import __version__

    return f"pdelab-{__version__}"
