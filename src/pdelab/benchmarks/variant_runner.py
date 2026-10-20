"""Single-variant benchmark runner.

Run as a *separate process* so that each variant gets a fresh CUDA
caching allocator and no other loaded model can contaminate its peak
memory measurement.

Usage:
    python -m pdelab.benchmarks.variant_runner \
        --variant fp32_eager --run-dir runs/<id> --out /tmp/v.json
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


def run_variant(
    variant: str,
    run_dir: Path,
    device_str: str = "cuda",
    batch_size: int = 1,
    n_blocks: int = 5,
    iters_per_block: int = 10,
    warmup: int = 3,
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

        def call():
            return rollout(model, hist, n_steps, stats, params=None, device=device)

    elif variant == "autocast_fp16":
        description = "torch.autocast(cuda, float16) around the same inference graph"

        def call():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return rollout(model, hist, n_steps, stats, params=None, device=device)

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

        def call():
            return rollout(compiled, hist, n_steps, stats, params=None, device=device)

        # Trigger compilation and measure the cold-start cost separately.
        call()
        _sync(device)
        compile_seconds = time.perf_counter() - t0

    else:
        raise ValueError(f"unknown variant {variant!r}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)

    out = call()
    timing = _time_blocks(call, device, n_blocks, iters_per_block, warmup)

    memory = {}
    if device.type == "cuda":
        memory = {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }

    return {
        "variant": variant,
        "description": description,
        "timing": timing,
        "memory": memory,
        "compile_seconds": compile_seconds,
        "throughput_samples_per_second": float(n_traj * n_steps / timing["median_seconds"]),
        "batch_size": int(n_traj),
        "rollout_steps": int(n_steps),
        "output_checksum": float(out.double().abs().mean().item()),
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
    )
    Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
