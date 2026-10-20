"""Sweep planning and execution.

A sweep matrix is declared in YAML. The planner expands it into an
explicit list of runs with an estimated cost, so a dry run can be
inspected before anything is executed. Execution is strictly sequential
and is guarded by the project job lock.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import config_hash, load_yaml, project_root, resolve_path
from .resources.preflight import HardwareProfile, JobLock, preflight
from .training.experiment import ExperimentRunner, load_experiment_config


def build_sweep(config_path: str | Path, profile: str = "prototype") -> dict[str, Any]:
    """Expand a sweep matrix into an explicit run list with estimates."""
    cfg_path = resolve_path(config_path)
    raw = load_yaml(cfg_path)
    matrix = raw.get("matrix", raw)

    systems: list[str] = list(matrix.get("systems", []))
    architectures: list[str] = list(matrix.get("architectures", []))
    conditions: list[str] = list(matrix.get("conditions", []))
    seeds: list[int] = list(matrix.get("seeds", [0]))
    base = matrix.get("base", {}) or {}
    profiles = matrix.get("profiles", {}) or {}
    prof = profiles.get(profile, {}) or {}
    applicable = [tuple(p) for p in matrix.get("applicable_pairs", [])]

    if profile == "full" and "seeds" not in matrix:
        seeds = [0, 1, 2]

    runs: list[dict[str, Any]] = []
    for system in systems:
        for arch in architectures:
            if applicable and (system, arch) not in applicable:
                continue
            for condition in conditions:
                for seed in seeds:
                    name = f"{system}_{arch}_{condition}_seed{seed}"
                    entry = {
                        "name": name,
                        "system": system,
                        "architecture": arch,
                        "condition": condition,
                        "seed": seed,
                        "epochs": prof.get("epochs", base.get("epochs", 20)),
                        "batch_size": prof.get(
                            "batch_size",
                            base.get("batch_size", 8 if system == "burgers" else 2),
                        ),
                    }
                    runs.append(entry)

    # Rough wall-clock estimate from the pilot measurement, if available.
    per_run_minutes = prof.get("estimated_minutes_per_run", 10.0)
    return {
        "sweep_config": str(cfg_path),
        "sweep_config_hash": config_hash(raw),
        "profile": profile,
        "n_runs": len(runs),
        "systems": systems,
        "architectures": architectures,
        "conditions": conditions,
        "seeds": seeds,
        "estimated_minutes_per_run": per_run_minutes,
        "estimated_total_minutes": round(len(runs) * per_run_minutes, 1),
        "runs": runs,
        "note": "dry run only; no training was executed by this command",
    }


def run_sweep(
    plan: dict[str, Any],
    dry_run: bool = False,
    hardware: str | Path | None = None,
) -> dict[str, Any]:
    """Execute a planned sweep sequentially."""
    if dry_run:
        return {**plan, "status": "dry_run", "executed": 0}

    hw = HardwareProfile.load(hardware or "configs/hardware/atul_laptop.yaml")
    base_dir = Path(plan["sweep_config"]).parent
    matrix = load_yaml(plan["sweep_config"]).get("matrix", {})

    results: list[dict[str, Any]] = []
    started = time.time()
    for entry in plan["runs"]:
        exp_name = f"{entry['system']}_{entry['architecture']}_{entry['condition']}_seed{entry['seed']}"
        exp_path = base_dir / f"{exp_name}.yaml"
        if not exp_path.exists():
            results.append({"name": exp_name, "status": "missing_config",
                            "path": str(exp_path)})
            continue

        lock = JobLock()
        if not lock.acquire(blocking=False):
            results.append({"name": exp_name, "status": "blocked_by_lock"})
            break
        try:
            cfg, raw = load_experiment_config(exp_path)
            cfg.seed = entry["seed"]
            cfg.epochs = entry["epochs"]
            cfg.batch_size = entry["batch_size"]
            runner = ExperimentRunner(cfg, raw, hw)
            manifest = runner.run()
            results.append({
                "name": exp_name,
                "status": manifest.get("status"),
                "run_id": manifest.get("run_id"),
                "best_val_loss": (manifest.get("training") or {}).get("best_val_loss"),
                "walltime_seconds": (manifest.get("training") or {}).get("walltime_seconds"),
            })
        finally:
            lock.release()

    return {
        "status": "completed",
        "profile": plan["profile"],
        "n_runs": len(plan["runs"]),
        "executed": len(results),
        "walltime_seconds": round(time.time() - started, 2),
        "results": results,
    }
