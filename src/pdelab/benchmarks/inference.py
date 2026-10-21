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

Accuracy validation
-------------------
The accuracy of a variant is decided by comparing the **full output
tensors** of the baseline and the variant on **identical inputs**:

* the maximum absolute error,
* the relative L2 error,
* the counts of non-finite values on both sides,
* the difference at **every** time step of the rollout, and
* the change of the rollout nRMSE against the reference solution.

The comparison runs over several batches of *development* (validation)
trajectories and over the **entire** rollout.  The scalar
``output_checksum`` is kept only as a supplementary *identity*
diagnostic: a sign flip or a spatial permutation has exactly the same
mean absolute value, so the checksum is never an accuracy proof.

The acceptance tolerance is **frozen before any candidate is
measured** and is never loosened afterwards; a candidate that fails it
is rejected and its numbers do not count towards a target.

Content identity
----------------
The artifact records the SHA-256 of the checkpoint, of the evaluated
dataset split, of the normalisation constants, of the evaluator
implementation, of the resolved benchmark settings and of the
protocol, so that the report can bind the numbers to the bytes that
produced them.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import project_root
from ..data.dataset import NormalizationStats, TrajectoryDataset, split_paths
from ..evaluation.metrics import (
    METRIC_IMPLEMENTATION_VERSION,
    canonical_json_sha256,
    evaluator_identity,
    normalization_identity,
    nrmse_per_trajectory,
    sha256_file,
    write_json_atomic,
)

VARIANTS = ["fp32_eager", "autocast_fp16", "compile_default", "compile_reduce_overhead"]

#: The protocol the evidence cohort declares for the artifacts of the
#: study.  The report binds an artifact to the cohort only when the
#: recorded protocol id matches the declared one, so the benchmark
#: artifact carries this id; the benchmark-specific protocol is
#: recorded next to it.
PROTOCOL_ID = "pdelab-free-rollout/2"

#: The benchmark-specific protocol id.
BENCHMARK_PROTOCOL_ID = "pdelab-inference-benchmark/1"

#: The split the accuracy validation runs on: the *development* split.
#: The test split is reserved for the reported metric.
CORRECTNESS_SPLIT = "val"

#: How long a single variant process may take before it is killed.
VARIANT_TIMEOUT_SECONDS = 3600.0


@dataclass(frozen=True)
class AccuracyTolerance:
    """The frozen acceptance tolerance of the inference benchmark.

    The tolerance is declared **before** any candidate is measured.  It
    is never loosened: a candidate that fails it is rejected, and the
    speed and memory numbers of a rejected candidate do not count
    towards a target.  Changing the tolerance means a new protocol id.
    """

    max_abs_error: float = 5.0e-2
    max_relative_l2: float = 1.0e-2
    max_rollout_nrmse_delta: float = 1.0e-2
    max_nonfinite_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: The frozen tolerance.  Do not edit it after a candidate has been
#: measured.
ACCURACY_TOLERANCE = AccuracyTolerance()

#: When the tolerance above was frozen (before any candidate was
#: measured).
TOLERANCE_FROZEN_UTC = "2026-10-21T00:00:00Z"


# ----------------------------------------------------------------------
# the element-wise comparison
# ----------------------------------------------------------------------
def compare_rollout_outputs(
    baseline: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray | None = None,
    tolerance: AccuracyTolerance = ACCURACY_TOLERANCE,
    label: str = "",
) -> dict[str, Any]:
    """Compare two rollout tensors element by element.

    ``baseline`` and ``candidate`` have shape ``(n_traj, n_steps,
    *spatial)``.  ``reference`` is the ground truth of the same shape;
    when it is given, the change of the rollout nRMSE is reported.

    A scalar checksum cannot do this: a sign flip or a spatial
    permutation has the same mean absolute value as the baseline and
    would pass a checksum comparison.
    """
    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if base.shape != cand.shape:
        return {
            "label": label,
            "passed": False,
            "failures": [
                f"the output shapes differ: baseline {base.shape}, "
                f"candidate {cand.shape}"
            ],
            "checks": {},
        }

    diff = cand - base
    max_abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0
    base_norm = float(np.linalg.norm(base))
    cand_norm = float(np.linalg.norm(cand))
    relative_l2 = (
        float(np.linalg.norm(diff) / base_norm) if base_norm > 0 else float("inf")
    )
    nonfinite_base = int(np.count_nonzero(~np.isfinite(base)))
    nonfinite_cand = int(np.count_nonzero(~np.isfinite(cand)))
    nonfinite_diff = int(np.count_nonzero(~np.isfinite(diff)))

    per_time_step: list[dict[str, Any]] = []
    for t in range(base.shape[1] if base.ndim > 1 else 0):
        d = diff[:, t]
        b = float(np.linalg.norm(base[:, t]))
        per_time_step.append({
            "step": int(t),
            "max_abs_error": float(np.max(np.abs(d))) if d.size else 0.0,
            "relative_l2": (float(np.linalg.norm(d)) / b) if b > 0 else float("inf"),
        })

    nrmse_base = nrmse_cand = nrmse_delta = None
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        if ref.shape == base.shape:
            nrmse_base = float(np.mean(nrmse_per_trajectory(base, ref)))
            nrmse_cand = float(np.mean(nrmse_per_trajectory(cand, ref)))
            nrmse_delta = float(nrmse_cand - nrmse_base)

    checks = {
        "max_abs_error": bool(max_abs_error <= tolerance.max_abs_error),
        "relative_l2": bool(relative_l2 <= tolerance.max_relative_l2),
        "nonfinite": bool(
            nonfinite_cand <= tolerance.max_nonfinite_count
            and nonfinite_diff <= tolerance.max_nonfinite_count
        ),
    }
    if nrmse_delta is not None:
        checks["rollout_nrmse_delta"] = bool(
            abs(nrmse_delta) <= tolerance.max_rollout_nrmse_delta
        )

    failures = [name for name, ok in checks.items() if not ok]
    worst_step = (
        int(max(per_time_step, key=lambda r: r["max_abs_error"])["step"])
        if per_time_step
        else None
    )
    return {
        "label": label,
        "shape": list(base.shape),
        "max_abs_error": max_abs_error,
        "relative_l2": relative_l2,
        "baseline_l2_norm": base_norm,
        "candidate_l2_norm": cand_norm,
        "nonfinite_baseline": nonfinite_base,
        "nonfinite_candidate": nonfinite_cand,
        "nonfinite_difference": nonfinite_diff,
        "per_time_step": per_time_step,
        "worst_time_step": worst_step,
        "nrmse_baseline": nrmse_base,
        "nrmse_candidate": nrmse_cand,
        "nrmse_delta": nrmse_delta,
        "checks": checks,
        "passed": not failures,
        "failures": failures,
    }


def legacy_scalar_checksum(output: np.ndarray) -> float:
    """The *old* accuracy check: a scalar mean absolute value.

    It is kept only as a historical reference and as a supplementary
    identity diagnostic.  It is **not** an accuracy measure: a sign flip
    or a spatial permutation has exactly the same value.
    """
    return float(np.asarray(output, dtype=np.float64).__abs__().mean())


def legacy_checksum_agreement(
    baseline: np.ndarray, candidate: np.ndarray, tolerance: float = 5e-2
) -> dict[str, Any]:
    """What the *old* check would say about two rollouts.

    Used by the correctness fixture to show what the scalar checksum
    misses.
    """
    a = legacy_scalar_checksum(baseline)
    b = legacy_scalar_checksum(candidate)
    rel = abs(b - a) / max(abs(a), 1e-12)
    return {
        "baseline_checksum": a,
        "candidate_checksum": b,
        "relative_difference": float(rel),
        "within_tolerance": bool(rel <= tolerance),
        "tolerance": float(tolerance),
        "role": "historical reference only; not an accuracy proof",
    }


# ----------------------------------------------------------------------
# the correctness workload
# ----------------------------------------------------------------------
def _build_correctness_workload(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    n_batches: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build the accuracy-validation workload from the development split.

    The workload holds several batches of trajectories, the **entire**
    rollout horizon, the initial history windows and the ground truth.
    The same file is handed to every variant, so the variants are
    compared on identical inputs.
    """
    cfg = manifest["experiment"]
    system = cfg["system"]
    paths = split_paths(cfg["dataset_dir"], system)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    stats = NormalizationStats.from_dict(ckpt["normalization"])
    history_len = int(cfg.get("history", 1))
    obs_times = manifest["dataset_manifest"]["system"]["observation_times"]
    n_steps = len(obs_times) - history_len

    split_path = paths[CORRECTNESS_SPLIT]
    if not split_path.exists():
        raise FileNotFoundError(
            f"the development split {split_path} does not exist; the accuracy "
            "validation cannot be run"
        )
    ds = TrajectoryDataset(split_path, stats, history=history_len, stride=1)
    n_avail = int(ds.n_traj)
    bs = max(1, min(batch_size, n_avail))
    nb = max(1, min(n_batches, n_avail // bs))
    total = nb * bs
    ids = np.linspace(0, n_avail - 1, total).round().astype(np.int64)

    spatial = ds.u.shape[2:]
    history = ds.u[ids][:, :history_len].astype(np.float32)
    reference = ds.u[ids][:, history_len : history_len + n_steps].astype(np.float32)
    workload = {
        "history": history.reshape(nb, bs, history_len, *spatial),
        "reference": reference.reshape(nb, bs, n_steps, *spatial),
        "trajectory_ids": ids.reshape(nb, bs),
    }
    info = {
        "split": CORRECTNESS_SPLIT,
        "path": str(split_path),
        "sha256": sha256_file(split_path),
        "n_batches": int(nb),
        "batch_size": int(bs),
        "n_trajectories": int(total),
        "trajectory_ids": ids.tolist(),
        "n_rollout_steps": int(n_steps),
        "history_length": int(history_len),
    }
    return workload, info


# ----------------------------------------------------------------------
# the isolated variant process
# ----------------------------------------------------------------------
def _run_variant_isolated(
    variant: str,
    run_dir: Path,
    device: str,
    batch_size: int,
    n_blocks: int,
    iters_per_block: int,
    tensor_dir: Path,
    correctness_inputs: Path | None,
) -> dict[str, Any]:
    """Measure one variant in a fresh process."""
    out_path = tensor_dir / "variant.json"
    cmd = [
        sys.executable, "-m", "pdelab.benchmarks.variant_runner",
        "--variant", variant,
        "--run-dir", str(run_dir),
        "--out", str(out_path),
        "--device", device,
        "--batch-size", str(batch_size),
        "--n-blocks", str(n_blocks),
        "--iters-per-block", str(iters_per_block),
        "--tensor-dir", str(tensor_dir),
    ]
    if correctness_inputs is not None:
        cmd += ["--correctness-inputs", str(correctness_inputs)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root()),
            check=False,
            timeout=VARIANT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "variant": variant,
            "status": "failed",
            "error": (
                f"the variant process exceeded the {VARIANT_TIMEOUT_SECONDS:.0f} s "
                "timeout and was terminated"
            ),
        }
    if proc.returncode != 0 or not out_path.exists():
        return {
            "variant": variant,
            "status": "failed",
            "error": (proc.stderr or proc.stdout)[-800:],
        }
    result = json.loads(out_path.read_text())
    result.setdefault("status", "completed")
    return result


# ----------------------------------------------------------------------
# the benchmark
# ----------------------------------------------------------------------
def run_benchmark(
    run_dir: str | Path,
    profile: str = "matched",
    device: str = "cuda",
    n_blocks: int = 5,
    iters_per_block: int = 10,
    correctness_batches: int = 3,
    correctness_batch_size: int = 2,
    preflight_report: dict[str, Any] | None = None,
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
        "split": "test",
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

    # ---- the accuracy-validation workload (identical for every variant) ----
    correctness_info: dict[str, Any]
    try:
        workload, correctness_info = _build_correctness_workload(
            run_dir,
            manifest,
            n_batches=correctness_batches,
            batch_size=correctness_batch_size,
        )
        correctness_error = None
    except Exception as exc:  # pragma: no cover - depends on the data on disk
        workload = None
        correctness_info = {
            "split": CORRECTNESS_SPLIT,
            "error": f"{type(exc).__name__}: {exc}",
        }
        correctness_error = correctness_info["error"]
    results["correctness_workload"] = correctness_info

    with tempfile.TemporaryDirectory(prefix="pdelab_bench_") as tmp:
        tmp_dir = Path(tmp)
        correctness_inputs = None
        if workload is not None:
            correctness_inputs = tmp_dir / "correctness_inputs.npz"
            np.savez_compressed(correctness_inputs, **workload)

        for bs in batch_sizes:
            for variant in VARIANTS:
                tensor_dir = tmp_dir / f"{variant}_bs{bs}"
                tensor_dir.mkdir(parents=True, exist_ok=True)
                started = time.time()
                m = _run_variant_isolated(
                    variant,
                    run_dir,
                    device,
                    bs,
                    n_blocks,
                    iters_per_block,
                    tensor_dir,
                    correctness_inputs,
                )
                m["walltime_seconds"] = round(time.time() - started, 2)
                m["batch_size"] = bs
                m["tensor_dir"] = str(tensor_dir)
                results["measurements"].append(m)

        # ---- the element-wise accuracy check of every variant ----
        results["accuracy"] = _accuracy_section(
            results["measurements"], workload, bs=batch_sizes[0]
        )
        for m in results["measurements"]:
            # the temporary directory is gone once the artifact is
            # written; it must not be recorded in the artifact.
            m.pop("tensor_dir", None)

    # ---- the content identity ----
    results["identity"] = _content_identity(
        run_dir, manifest, profile=profile, device=device,
        n_blocks=n_blocks, iters_per_block=iters_per_block,
        batch_sizes=batch_sizes, n_steps=n_steps, grid=grid,
        history_len=history_len, correctness_info=correctness_info,
        preflight_report=preflight_report,
    )
    results["protocol"] = {
        "id": PROTOCOL_ID,
        "benchmark_protocol_id": BENCHMARK_PROTOCOL_ID,
        "type": "matched_inference_benchmark",
        "split": "test",
        "accuracy_split": CORRECTNESS_SPLIT,
        "observation_dt": (
            float(obs_times[1] - obs_times[0]) if len(obs_times) > 1 else None
        ),
        "n_rollout_steps": int(n_steps),
        "history_length": history_len,
        "residual_evaluators": [],
        "residual_support": "not applicable (inference benchmark)",
    }
    if correctness_error:
        results["accuracy"]["status"] = "failed"
        results["accuracy"]["error"] = correctness_error
    if results["accuracy"].get("status") == "failed":
        # The timings may exist, but a benchmark whose accuracy could
        # not be validated does not produce a bindable result.
        results["status"] = "failed"
        results["status_reason"] = (
            "the accuracy validation could not be completed: "
            + str(results["accuracy"].get("error", "unknown reason"))
        )

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
    results["accuracy_note"] = (
        "The accuracy of a variant is decided by the element-wise comparison of "
        "the full output tensors against the FP32 baseline on identical inputs "
        f"(tolerance frozen on {TOLERANCE_FROZEN_UTC}: "
        f"{ACCURACY_TOLERANCE.to_dict()}). The scalar output_checksum is a "
        "supplementary identity diagnostic only and is never an accuracy proof."
    )

    out_path = run_dir / f"benchmark_{profile}.json"
    digest = write_json_atomic(out_path, results)
    results["artifact_sha256"] = digest
    results["benchmark_path"] = str(out_path)
    return results


def _accuracy_section(
    measurements: list[dict[str, Any]],
    workload: dict[str, np.ndarray] | None,
    *,
    bs: int,
) -> dict[str, Any]:
    """Compare every variant with the FP32 baseline, element by element."""
    section: dict[str, Any] = {
        "method": (
            "the full output tensors of the baseline and of the variant are "
            "compared element by element on identical inputs, over the entire "
            "rollout and over several development batches"
        ),
        "tolerance": ACCURACY_TOLERANCE.to_dict(),
        "tolerance_frozen_utc": TOLERANCE_FROZEN_UTC,
        "status": "completed",
        "variants": {},
    }
    if workload is None:
        section["status"] = "not_run"
        return section

    base = next(
        (m for m in measurements
         if m.get("variant") == "fp32_eager" and m.get("batch_size") == bs
         and m.get("status") != "failed"),
        None,
    )
    if base is None:
        section["status"] = "failed"
        section["error"] = "the FP32 baseline did not complete"
        return section
    base_dir = Path(base["tensor_dir"])
    base_correctness = base_dir / "correctness_fp32_eager.npz"
    base_timed = base_dir / "timed_fp32_eager.npy"
    if not base_correctness.exists():
        section["status"] = "failed"
        section["error"] = "the baseline did not write its output tensors"
        return section
    base_rollouts = np.load(base_correctness)["rollouts"]
    reference = workload["reference"]

    for m in measurements:
        if m.get("batch_size") != bs or m.get("status") == "failed":
            continue
        if m["variant"] == "fp32_eager":
            m["accuracy_within_tolerance"] = True
            m["accuracy"] = {
                "role": "baseline (the reference the variants are compared to)",
                "passed": True,
            }
            continue
        cand_dir = Path(m["tensor_dir"])
        cand_path = cand_dir / f"correctness_{m['variant']}.npz"
        if not cand_path.exists():
            m["accuracy_within_tolerance"] = False
            m["accuracy"] = {
                "passed": False,
                "failures": ["the variant did not write its output tensors"],
            }
            continue
        cand_rollouts = np.load(cand_path)["rollouts"]
        batch_results = []
        for b in range(base_rollouts.shape[0]):
            batch_results.append(
                compare_rollout_outputs(
                    base_rollouts[b],
                    cand_rollouts[b],
                    reference=reference[b],
                    tolerance=ACCURACY_TOLERANCE,
                    label=f"batch{b}",
                )
            )
        # the timed workload is compared as well
        timed = None
        cand_timed_path = cand_dir / f"timed_{m['variant']}.npy"
        if base_timed.exists() and cand_timed_path.exists():
            timed = compare_rollout_outputs(
                np.load(base_timed),
                np.load(cand_timed_path),
                tolerance=ACCURACY_TOLERANCE,
                label="timed_workload",
            )
        failures: list[str] = []
        for r in batch_results:
            failures.extend(f"{r['label']}: {f}" for f in r["failures"])
        if timed is not None:
            failures.extend(f"timed_workload: {f}" for f in timed["failures"])
        worst = max(
            batch_results,
            key=lambda r: r["max_abs_error"],
            default=None,
        )
        m["accuracy_within_tolerance"] = bool(not failures)
        m["accuracy"] = {
            "batches": batch_results,
            "timed_workload": timed,
            "worst_batch": worst["label"] if worst else None,
            "max_abs_error": worst["max_abs_error"] if worst else None,
            "relative_l2": worst["relative_l2"] if worst else None,
            "nrmse_delta": worst["nrmse_delta"] if worst else None,
            "passed": bool(not failures),
            "failures": failures,
            "tolerance": ACCURACY_TOLERANCE.to_dict(),
        }
        section["variants"][m["variant"]] = {
            "passed": m["accuracy"]["passed"],
            "max_abs_error": m["accuracy"]["max_abs_error"],
            "relative_l2": m["accuracy"]["relative_l2"],
            "nrmse_delta": m["accuracy"]["nrmse_delta"],
            "failures": failures,
        }
    return section


def _content_identity(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    profile: str,
    device: str,
    n_blocks: int,
    iters_per_block: int,
    batch_sizes: list[int],
    n_steps: int,
    grid: list[int],
    history_len: int,
    correctness_info: dict[str, Any],
    preflight_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """The content identity the report binds the artifact to."""
    cfg = manifest["experiment"]
    system = cfg["system"]
    paths = split_paths(cfg["dataset_dir"], system)
    ckpt_path = run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = NormalizationStats.from_dict(ckpt["normalization"])
    split_path = paths["test"]
    manifest_files = (manifest.get("dataset_manifest") or {}).get("files") or {}

    resolved = {
        "metric_implementation_version": METRIC_IMPLEMENTATION_VERSION,
        "benchmark_protocol_id": BENCHMARK_PROTOCOL_ID,
        "profile": profile,
        "device": device,
        "batch_sizes": list(batch_sizes),
        "matched_batch_size": int(batch_sizes[0]),
        "n_blocks": int(n_blocks),
        "iters_per_block": int(iters_per_block),
        "variants": list(VARIANTS),
        "rollout_steps": int(n_steps),
        "grid": list(grid),
        "history_length": int(history_len),
        "system": system,
        "architecture": cfg["architecture"],
        "accuracy_split": CORRECTNESS_SPLIT,
        "accuracy_n_batches": correctness_info.get("n_batches"),
        "accuracy_batch_size": correctness_info.get("batch_size"),
        "accuracy_n_trajectories": correctness_info.get("n_trajectories"),
        "accuracy_tolerance": ACCURACY_TOLERANCE.to_dict(),
        "accuracy_tolerance_frozen_utc": TOLERANCE_FROZEN_UTC,
        "isolation": "each variant measured in a separate process",
    }
    protocol = {
        "id": PROTOCOL_ID,
        "benchmark": {
            "id": BENCHMARK_PROTOCOL_ID,
            "variants": list(VARIANTS),
            "isolation": "each variant measured in a separate process",
            "accuracy_split": CORRECTNESS_SPLIT,
            "accuracy_n_batches": correctness_info.get("n_batches"),
            "accuracy_batch_size": correctness_info.get("batch_size"),
            "accuracy_trajectory_ids": correctness_info.get("trajectory_ids"),
            "accuracy_n_rollout_steps": int(n_steps),
            "tolerance": ACCURACY_TOLERANCE.to_dict(),
            "tolerance_frozen_utc": TOLERANCE_FROZEN_UTC,
        },
    }
    protocol["sha256"] = canonical_json_sha256(
        {"id": protocol["id"], "benchmark": protocol["benchmark"]}
    )
    identity = {
        "evaluator": evaluator_identity(),
        "protocol": protocol,
        "resolved_settings": resolved,
        "resolved_settings_sha256": canonical_json_sha256(resolved),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "dataset": {
            "split": "test",
            "path": str(split_path),
            "sha256": sha256_file(split_path) if split_path.exists() else None,
            "manifest_sha256": (manifest_files.get("test") or {}).get("sha256"),
        },
        "normalization": normalization_identity(stats),
    }
    if preflight_report is not None:
        identity["preflight"] = {
            "timestamp_utc": preflight_report.get("timestamp_utc"),
            "ok": preflight_report.get("ok"),
            "violations": preflight_report.get("violations"),
            "warnings": preflight_report.get("warnings"),
            "project_lease": (
                "the project job lock (runs/.pdelab.lock) was held for the whole "
                "benchmark; it excludes cooperating PDELab jobs only and is not "
                "a system-wide scheduler"
            ),
        }
    return identity
