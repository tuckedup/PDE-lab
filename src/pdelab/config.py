"""Validated YAML configuration loading.

Configurations are plain YAML files under ``configs/``. Every loader
returns a validated, immutable-ish dataclass and records the SHA-256
hash of the source file so a run manifest can pin exactly which
configuration produced a result.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Locate the repository root by walking up from this file."""
    env = os.environ.get("PDELAB_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            return parent
    return here.parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config {p} did not parse to a mapping")
    return data


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(data: dict[str, Any]) -> str:
    """Stable hash of a config dictionary (independent of key order)."""
    blob = yaml.safe_dump(data, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch ``a.b.c`` from nested dicts."""
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass
class PDESystemConfig:
    """Declarative description of a benchmark PDE system."""

    name: str
    dimension: int
    solver: str
    solver_grid: tuple[int, ...]
    observation_grid: tuple[int, ...]
    observation_times: list[float]
    t_final: float
    params: dict[str, float] = field(default_factory=dict)
    initial_condition: dict[str, Any] = field(default_factory=dict)
    reference_solver: str = ""
    reference_grid: tuple[int, ...] = ()
    restriction: str = "spectral_truncate_then_subsample"
    source_path: str = ""
    source_hash: str = ""

    @staticmethod
    def from_dict(data: dict[str, Any], source: str = "", source_hash: str = "") -> "PDESystemConfig":
        return PDESystemConfig(
            name=data["name"],
            dimension=int(data["dimension"]),
            solver=data["solver"],
            solver_grid=tuple(data["solver_grid"]),
            observation_grid=tuple(data["observation_grid"]),
            observation_times=[float(t) for t in data["observation_times"]],
            t_final=float(data["t_final"]),
            params={k: float(v) for k, v in (data.get("params") or {}).items()},
            initial_condition=data.get("initial_condition") or {},
            reference_solver=data.get("reference_solver", ""),
            reference_grid=tuple(data.get("reference_grid") or data["solver_grid"]),
            restriction=data.get("restriction", "spectral_truncate_then_subsample"),
            source_path=source,
            source_hash=source_hash,
        )
