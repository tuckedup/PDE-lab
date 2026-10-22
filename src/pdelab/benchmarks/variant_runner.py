"""Single-variant benchmark runner.

Run as a *separate process* so that each variant gets a fresh CUDA
caching allocator and no other loaded model can contaminate its peak
memory measurement.

Usage:
    python -m pdelab.benchmarks.variant_runner \
        --variant fp32_eager --run-dir runs/<id> --out /tmp/v.json

Outputs of the variant
----------------------
Besides the timing/memory report (written to ``--out``), the runner
writes the **full rollout tensors** so that the parent process can
compare the variants element by element:

``<tensor-dir>/timed_<variant>.npy``
    the rollout of the timed workload (the reported workload),
``<tensor-dir>/correctness_<variant>.npz``
    the rollouts of the correctness workload (several development
    batches, the entire rollout), computed on inputs that are written
    once by the parent and therefore *identical* for every variant.

The scalar ``output_checksum`` is kept only as a supplementary
*identity* diagnostic (it says "this process produced the same numbers
as the reference process").  It is **not** an accuracy measure: a sign
flip or a spatial permutation has exactly the same mean absolute
value.  The accuracy verdict is computed by the parent from the full
tensors.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import project_root
from ..data.dataset import NormalizationStats, TrajectoryDataset, split_paths
from ..evaluation.rollout import rollout
from ..models.registry import build_model_from_config


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_blocks(
    fn, device: torch.device, n_blocks: int, iters_per_block: int, warmup: int
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    _sync(device)
    block_times: list[float] = []
    for _ in range(n_blocks):
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters_per_block):
            fn()
        _sync(device)
        block_times.append((time.perf_counter() - t0) / iters_per_block)
    return {
        "median_seconds": statistics.median(block_times),
        "min_seconds": min(block_times),
        "max_seconds": max(block_times),
        "stdev_seconds": statistics.pstdev(block_times) if len(block_times) > 1 else 0.0,
        "blocks": block_times,
        "iters_per_block": iters_per_block,
        "n_blocks": n_blocks,
    }


def load_model_and_workload(run_dir: Path, device: torch.device, batch_size: int):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["experiment"]
    system = cfg["system"]
    paths = split_paths(cfg["dataset_dir"], system)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    stats = NormalizationStats.from_dict(ckpt["normalization"])
    model = build_model_from_config(
        {
            "architecture": cfg["architecture"],
            "history": cfg.get("history", 1),
            **{k: v for k, v in (cfg.get("model_kwargs") or {}).items()},
        },
        dimension=2 if system == "allen_cahn" else 1,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    history_len = int(cfg.get("history", 1))
    obs_times = manifest["dataset_manifest"]["system"]["observation_times"]
    n_steps = len(obs_times) - history_len
    test_set = TrajectoryDataset(paths["test"], stats, history=history_len, stride=1)
    n_traj = min(batch_size, test_set.n_traj)
    hist = torch.from_numpy(test_set.u[:n_traj, :history_len]).to(device)
    return manifest, model, stats, hist, n_steps, n_traj


def _drop_channel(out: torch.Tensor) -> torch.Tensor:
    """``(B, T, 1, *spatial)`` -> ``(B, T, *spatial)``."""
    if out.dim() >= 3 and out.shape[2] == 1:
        return out[:, :, 0]
    return out


def _correctness_rollouts(
    run,
    inputs: dict[str, np.ndarray],
    n_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Roll out the correctness workload, batch by batch.

    The inputs are the *same file* for every variant, so the variants
    are compared on identical inputs.  The rollout goes through the
    **same** variant wrapper as the timed workload (autocast context,
    compiled model, ...), otherwise the accuracy check would compare
    the baseline against itself.
    """
    history = inputs["history"]
    outputs = []
    with torch.no_grad():
        for b in range(history.shape[0]):
            hist = torch.from_numpy(np.ascontiguousarray(history[b])).to(device)
            out = run(hist)
            outputs.append(_drop_channel(out).detach().cpu().numpy())
    return np.stack(outputs, axis=0)


def run_variant(
    variant: str,
    run_dir: Path,
    device_str: str = "cuda",
    batch_size: int = 1,
    n_blocks: int = 5,
    iters_per_block: int = 10,
    warmup: int = 3,
    tensor_dir: Path | None = None,
    correctness_inputs: Path | None = None,
) -> dict[str, Any]:
    device = torch.device(device_str)
    torch.set_grad_enabled(False)

    manifest, model, stats, hist, n_steps, n_traj = load_model_and_workload(
        run_dir, device, batch_size
    )

    compile_seconds = 0.0
    description = ""
    if variant == "fp32_eager":
        description = "model.eval(), inference-only, weights resident on device"

        def run(h):
            return rollout(model, h, n_steps, stats, params=None, device=device)

    elif variant == "autocast_fp16":
        description = "torch.autocast(cuda, float16) around the same inference graph"

        def run(h):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return rollout(model, h, n_steps, stats, params=None, device=device)

    elif variant in ("compile_default", "compile_reduce_overhead"):
        mode = "default" if variant == "compile_default" else "reduce-overhead"
        description = f"torch.compile(model, mode={mode!r})"
        if mode == "reduce-overhead":
            # The rollout produces user-visible outputs that persist across
            # graph generations, so the trees must clone them.
            import torch._inductor.config as inductor_config

            inductor_config.triton.cudagraph_trees_generation_cloning = "user_visible"
        t0 = time.perf_counter()
        compiled = torch.compile(model, mode=mode)

        def run(h):
            return rollout(compiled, h, n_steps, stats, params=None, device=device)

        # Trigger compilation and measure the cold-start cost separately.
        run(hist)
        _sync(device)
        compile_seconds = time.perf_counter() - t0

    else:
        raise ValueError(f"unknown variant {variant!r}")

    def call():
        return run(hist)

    # ---- uniform warm-up BEFORE the reset point -------------------
    # The warm-up must precede the reset so that the measurement covers
    # the *steady state* of the variant. This matters for the
    # CUDA-graphs variants: their private memory pool is allocated when
    # the graph is captured, and a reset taken before that would
    # charge the pool to the measured region and hide it.
    for _ in range(warmup):
        call()
    _sync(device)

    # ---- reset point ------------------------------------------------
    # Everything that is *persistent* for the variant (the model
    # weights, the CUDA-graph private pool, the resident input) is
    # already in place at this point, and the counters are reset so
    # that the reported peak describes the steady-state working set.
    memory_before: dict[str, Any] = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory_before = {
            "allocated_gib": torch.cuda.memory_allocated(device) / 1024**3,
            "reserved_gib": torch.cuda.memory_reserved(device) / 1024**3,
        }
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)

    out = call()
    timing = _time_blocks(call, device, n_blocks, iters_per_block, warmup=0)

    memory: dict[str, Any] = {}
    if device.type == "cuda":
        free_after, total = torch.cuda.mem_get_info(device)
        weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        input_bytes = hist.numel() * hist.element_size()
        output_bytes = out.numel() * out.element_size()
        memory = {
            # The declared metric: peak *allocated* during the measured
            # region. This is NOT total device memory.
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            # What was already resident when the counters were reset:
            # the weights and, for the CUDA-graph variants, the graph
            # private pool.
            "allocated_before_reset_gib": memory_before.get("allocated_gib"),
            "reserved_before_reset_gib": memory_before.get("reserved_gib"),
            # Persistent bytes that every variant must hold: the model
            # weights, the resident input and one rollout output.
            "persistent_weights_bytes": int(weights_bytes),
            "persistent_input_bytes": int(input_bytes),
            "persistent_output_bytes": int(output_bytes),
            "persistent_total_bytes": int(weights_bytes + input_bytes + output_bytes),
            # Device-level observation: free memory on the device now.
            "device_free_gib": free_after / 1024**3,
            "device_total_gib": total / 1024**3,
            "note": (
                "peak_allocated is the declared metric and is not total device "
                "memory; compare it with peak_reserved and with the "
                "persistent_* byte counts before drawing a device-memory "
                "conclusion"
            ),
        }

    # ---- the full output tensors, for the element-wise comparison ----
    tensor_paths: dict[str, str] = {}
    if tensor_dir is not None:
        tensor_dir = Path(tensor_dir)
        tensor_dir.mkdir(parents=True, exist_ok=True)
        timed = _drop_channel(out).detach().cpu().numpy()
        timed_path = tensor_dir / f"timed_{variant}.npy"
        np.save(timed_path, timed)
        tensor_paths["timed"] = str(timed_path)
        if correctness_inputs is not None:
            inputs = dict(np.load(correctness_inputs))
            rollouts = _correctness_rollouts(run, inputs, n_steps, device)
            corr_path = tensor_dir / f"correctness_{variant}.npz"
            np.savez_compressed(corr_path, rollouts=rollouts)
            tensor_paths["correctness"] = str(corr_path)

    return {
        "variant": variant,
        "description": description,
        "timing": timing,
        "memory": memory,
        "compile_seconds": compile_seconds,
        "throughput_samples_per_second": float(n_traj * n_steps / timing["median_seconds"]),
        "batch_size": int(n_traj),
        "rollout_steps": int(n_steps),
        "output_shape": list(_drop_channel(out).shape),
        "output_tensors": tensor_paths,
        # Supplementary *identity* diagnostic only.  Two completely
        # different trajectories share this number, so it is never used
        # as an accuracy proof.
        "output_checksum": float(out.double().abs().mean().item()),
        "output_checksum_role": (
            "supplementary identity diagnostic (mean absolute value of the "
            "rollout); NOT an accuracy measure"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--n-blocks", type=int, default=5)
    ap.add_argument("--iters-per-block", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tensor-dir", default=None,
                    help="directory in which the full rollout tensors are written")
    ap.add_argument("--correctness-inputs", default=None,
                    help="npz file with the correctness workload (identical inputs "
                         "for every variant)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = project_root() / run_dir
    result = run_variant(
        args.variant,
        run_dir,
        device_str=args.device,
        batch_size=args.batch_size,
        n_blocks=args.n_blocks,
        iters_per_block=args.iters_per_block,
        warmup=args.warmup,
        tensor_dir=Path(args.tensor_dir) if args.tensor_dir else None,
        correctness_inputs=Path(args.correctness_inputs) if args.correctness_inputs else None,
    )
    Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
