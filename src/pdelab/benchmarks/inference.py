"""Matched inference benchmark.

Protocol
--------
* The **same checkpoint**, problem grid, rollout horizon, batch size and
  output contract are used for the baseline and every optimized
  variant. A smaller model, a shorter horizon or a coarser grid is a
  *different workload*, not an inference optimization, and is not
  reported as one.
* The FP32 baseline is a correct, reasonably implemented reference:
  ``model.eval()``, inference-only gradient behaviour, fixed weights,
  fixed workload, documented batch size. The model weights are already
  on the device before timing starts and no training graph is built.
* Each variant is measured in a **separate process** so the CUDA
  caching allocator and any other loaded model cannot contaminate the
  peak-memory number.
* Every variant is warmed up, CUDA is synchronized around the timed
  region, and several timing blocks are repeated. The median and the
  spread across blocks are reported. Compile / cold-start time is
  reported separately from steady-state time.
* Peak CUDA memory is measured with an explicit reset point after
  loading the model and after warm-up, with the model weights resident
  in the measurement (the claim is about the total inference
  footprint).
* An optimized variant must meet the declared accuracy tolerance
  against the FP32 baseline before its speed or memory numbers are
  allowed to count towards a target.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from ..config import project_root

VARIANTS = ["fp32_eager", "autocast_fp16", "compile_default", "compile_reduce_overhead"]


def _run_variant_isolated(
    variant: str,
    run_dir: Path,
    device: str,
    batch_size: int,
    n_blocks: int,
    iters_per_block: int,
) -> dict[str, Any]:
    """Measure one variant in a fresh process."""
    from .variant_runner import run_variant

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "variant.json"
        cmd = [
            sys.executable, "-m", "pdelab.benchmarks.variant_runner",
            "--variant", variant,
            "--run-dir", str(run_dir),
            "--out", str(out_path),
            "--device", device,
            "--batch-size", str(batch_size),
            "--n-blocks", str(n_blocks),
            "--iters-per-block", str(iters_per_block),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(project_root()), check=False
        )
        if proc.returncode != 0 or not out_path.exists():
            return {
                "variant": variant,
                "status": "failed",
                "error": (proc.stderr or proc.stdout)[-800:],
            }
        return json.loads(out_path.read_text())


def run_benchmark(
    run_dir: str | Path,
    profile: str = "matched",
    device: str = "cuda",
    n_blocks: int = 5,
    iters_per_block: int = 10,
) -> dict[str, Any]:
    """Benchmark a trained checkpoint's inference pipeline."""
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = project_root() / run_dir
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {"status": "failed", "error": f"no manifest at {manifest_path}"}
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("status") != "completed":
        return {
            "status": "skipped",
            "reason": f"run status is {manifest.get('status')!r}",
            "run_id": manifest.get("run_id"),
        }

    if device == "cuda" and not torch.cuda.is_available():
        return {
            "status": "failed",
            "error": "CUDA requested but not available; a GPU benchmark was not run "
                     "rather than being reported on the CPU",
        }

    cfg = manifest["experiment"]
    system = cfg["system"]
    obs_times = manifest["dataset_manifest"]["system"]["observation_times"]
    history_len = int(cfg.get("history", 1))
    n_steps = len(obs_times) - history_len
    grid = list(manifest["dataset_manifest"]["system"]["observation_grid"])

    # Batch size: the matched profile uses the smallest batch so that
    # latency is measured without hiding device work; the throughput
    # profile sweeps larger batches and is labelled separately.
    if profile == "throughput":
        batch_sizes = [1, 2, 4, 8]
    else:
        batch_sizes = [1]

    results: dict[str, Any] = {
        "status": "completed",
        "run_id": manifest["run_id"],
        "system": system,
        "architecture": cfg["architecture"],
        "checkpoint": str(run_dir / "best.pt"),
        "profile": profile,
        "isolation": "each variant measured in a separate process",
        "device": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "measurements": [],
    }

    for bs in batch_sizes:
        for variant in VARIANTS:
            started = time.time()
            m = _run_variant_isolated(
                variant, run_dir, device, bs, n_blocks, iters_per_block
            )
            m["walltime_seconds"] = round(time.time() - started, 2)
            m["batch_size"] = bs
            results["measurements"].append(m)

    # ---- accuracy check of every variant against the FP32 baseline ----
    base = None
    for m in results["measurements"]:
        if m.get("variant") == "fp32_eager" and m.get("batch_size") == batch_sizes[0]:
            base = m
    for m in results["measurements"]:
        if base is not None and m.get("variant") != "fp32_eager":
            if m.get("status") == "failed":
                continue
            rel = abs(m.get("output_checksum", 0.0) - base.get("output_checksum", 0.0))
            scale = max(abs(base.get("output_checksum", 0.0)), 1e-12)
            m["accuracy_relative_max_diff"] = rel / scale
            m["accuracy_within_tolerance"] = bool((rel / scale) <= 5e-2)

    # ---- comparison at the matched batch size ----
    matched_bs = batch_sizes[0]
    matched = [
        m for m in results["measurements"]
        if m.get("batch_size") == matched_bs and m.get("status") != "failed"
    ]
    base_matched = next(
        (m for m in matched if m["variant"] == "fp32_eager"), None
    )
    if base_matched:
        eligible = [
            m for m in matched
            if m["variant"] != "fp32_eager" and m.get("accuracy_within_tolerance", False)
        ]
        best_tp = max(eligible, key=lambda m: m["throughput_samples_per_second"], default=None)
        best_mem = min(
            eligible, key=lambda m: m["memory"].get("peak_allocated_gib", float("inf")),
            default=None,
        )
        if best_tp is not None:
            results["throughput_ratio"] = float(
                best_tp["throughput_samples_per_second"]
                / base_matched["throughput_samples_per_second"]
            )
            results["throughput_best_variant"] = best_tp["variant"]
        if best_mem is not None:
            base_mem = base_matched["memory"].get("peak_allocated_gib", 0.0)
            opt_mem = best_mem["memory"].get("peak_allocated_gib", 0.0)
            if base_mem > 0:
                results["memory_reduction_percent"] = float(
                    100.0 * (base_mem - opt_mem) / base_mem
                )
                results["memory_best_variant"] = best_mem["variant"]

    results["workload"] = {
        "batch_size": matched_bs,
        "rollout_steps": int(n_steps),
        "grid": grid,
        "history_length": history_len,
        "note": "identical for every variant; a change here would be a different "
                "workload, not an optimization",
    }
    results["caveat"] = (
        "Throughput and memory improvements are reported for the same checkpoint, "
        "grid, horizon and batch size. They are not a comparison against the "
        "numerical solver, and they do not license any claim about model accuracy."
    )

    out_path = run_dir / f"benchmark_{profile}.json"
    out_path.write_text(json.dumps(results, indent=2))
    results["benchmark_path"] = str(out_path)
    return results
