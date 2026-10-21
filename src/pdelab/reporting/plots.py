"""Inspectable plots generated from stored artifacts.

Every figure is produced from a run's stored manifest / evaluation /
benchmark file, so the figures cannot drift from the measurements. Each
figure records the dataset, grid, horizon, checkpoint and units in its
title or caption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import project_root
from ..data.dataset import NormalizationStats, TrajectoryDataset, split_paths
from ..evaluation.rollout import rollout
from ..models.registry import build_model_from_config

RUNS = project_root() / "runs"
RESULTS = project_root() / "results"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _usable_evaluation(path: Path) -> dict[str, Any] | None:
    """The evaluation of a run, if it is usable for a figure.

    A figure must not draw a stale or protocol-invalid number, so an
    artifact that does not carry a content identity (the pre-identity
    protocol) is not plotted.
    """
    ev = _load(path)
    if not ev or ev.get("status") != "completed":
        return None
    identity = ev.get("identity")
    if not isinstance(identity, dict):
        return None
    return ev


def _model_for(run_dir: Path, manifest: dict, device="cpu"):
    cfg = manifest["experiment"]
    ckpt = torch_load(run_dir / "best.pt", device)
    model = build_model_from_config(
        {
            "architecture": cfg["architecture"],
            "history": cfg.get("history", 1),
            **{k: v for k, v in (cfg.get("model_kwargs") or {}).items()},
        },
        dimension=2 if cfg["system"] == "allen_cahn" else 1,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def torch_load(path: Path, device="cpu"):
    import torch

    return torch.load(path, map_location=device, weights_only=False)


def plot_1d_space_time(run_id: str, out_dir: Path | None = None) -> list[str]:
    """Reference / prediction / error space-time heatmaps for 1D."""
    import torch

    run_dir = RUNS / run_id
    manifest = _load(run_dir / "manifest.json")
    evaluation = _usable_evaluation(run_dir / "evaluation.json")
    if manifest is None or evaluation is None:
        return []
    if manifest["experiment"]["system"] != "burgers":
        return []

    out_dir = out_dir or (RESULTS / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = manifest["experiment"]
    system = cfg["system"]
    paths = split_paths(cfg["dataset_dir"], system)
    model, ckpt = _model_for(run_dir, manifest)
    stats = NormalizationStats.from_dict(ckpt["normalization"])
    hist_len = int(cfg.get("history", 1))
    obs_times = manifest["dataset_manifest"]["system"]["observation_times"]
    n_steps = len(obs_times) - hist_len

    ds = TrajectoryDataset(paths["test"], stats, history=hist_len)
    t = 0
    h = torch.from_numpy(ds.u[t, :hist_len]).unsqueeze(0)
    pred = rollout(model, h, n_steps, stats, params=None, device="cpu")[0, :, 0].numpy()
    ref = ds.u[t, hist_len : hist_len + n_steps]
    err = np.abs(pred - ref)
    times = np.asarray(obs_times[hist_len:])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, data, title, cmap in [
        (axes[0], ref, "reference", "viridis"),
        (axes[1], pred, "prediction (free rollout)", "viridis"),
        (axes[2], err, "|error|", "magma"),
    ]:
        im = ax.imshow(
            data, aspect="auto", origin="lower", cmap=cmap,
            extent=[times[0], times[-1], 0, data.shape[-1]],
        )
        ax.set_xlabel("t")
        ax.set_ylabel("x (grid index)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="u" if cmap == "viridis" else "|du|")
    fig.suptitle(
        f"{run_id} - 1D Burgers space-time | grid {list(manifest['dataset_manifest']['system']['observation_grid'])}, "
        f"dt={times[1]-times[0]:g}, checkpoint best.pt (epoch {ckpt.get('epoch')})"
    )
    path = out_dir / f"{run_id}_space_time.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [str(path)]


def plot_2d_snapshots(run_id: str, out_dir: Path | None = None) -> list[str]:
    """Reference / prediction / error snapshot sequences for 2D."""
    import torch

    run_dir = RUNS / run_id
    manifest = _load(run_dir / "manifest.json")
    evaluation = _usable_evaluation(run_dir / "evaluation.json")
    if manifest is None or evaluation is None:
        return []
    if manifest["experiment"]["system"] != "allen_cahn":
        return []

    out_dir = out_dir or (RESULTS / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = manifest["experiment"]
    paths = split_paths(cfg["dataset_dir"], cfg["system"])
    model, ckpt = _model_for(run_dir, manifest)
    stats = NormalizationStats.from_dict(ckpt["normalization"])
    hist_len = int(cfg.get("history", 1))
    obs_times = manifest["dataset_manifest"]["system"]["observation_times"]
    n_steps = len(obs_times) - hist_len

    ds = TrajectoryDataset(paths["test"], stats, history=hist_len)
    h = torch.from_numpy(ds.u[0, :hist_len]).unsqueeze(0)
    pred = rollout(model, h, n_steps, stats, params=None, device="cpu")[0, :, 0].numpy()
    ref = ds.u[0, hist_len : hist_len + n_steps]

    picks = np.linspace(0, n_steps - 1, 5).astype(int)
    fig, axes = plt.subplots(3, len(picks), figsize=(3 * len(picks), 9.5),
                             constrained_layout=True)
    for j, k in enumerate(picks):
        for row, (data, title) in enumerate([
            (ref[k], f"reference t={obs_times[hist_len+k]:g}"),
            (pred[k], f"prediction t={obs_times[hist_len+k]:g}"),
            (np.abs(pred[k] - ref[k]), f"|error| t={obs_times[hist_len+k]:g}"),
        ]):
            im = axes[row, j].imshow(data, origin="lower", cmap="viridis")
            axes[row, j].set_title(title, fontsize=9)
            axes[row, j].set_xticks([])
            axes[row, j].set_yticks([])
            fig.colorbar(im, ax=axes[row, j], fraction=0.046)
    fig.suptitle(
        f"{run_id} - 2D Allen-Cahn snapshots | grid "
        f"{list(manifest['dataset_manifest']['system']['observation_grid'])}, "
        f"dt={obs_times[1]-obs_times[0]:g}"
    )
    path = out_dir / f"{run_id}_snapshots.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [str(path)]


def plot_learning_curves(out_dir: Path | None = None) -> list[str]:
    """Training and validation loss curves for every run."""
    out_dir = out_dir or (RESULTS / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [d for d in sorted(RUNS.iterdir()) if (d / "manifest.json").exists()]
    if not runs:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for d in runs:
        m = _load(d / "manifest.json")
        if not m or m.get("status") != "completed":
            continue
        hist = (m.get("training") or {}).get("history") or []
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        tr = [h["train"].get("data", np.nan) for h in hist]
        va = [h["val"].get("data", np.nan) for h in hist]
        axes[0].plot(epochs, tr, label=d.name, lw=1.0)
        axes[1].plot(epochs, va, label=d.name, lw=1.0)
    axes[0].set_title("training data loss (normalized MSE)")
    axes[1].set_title("validation data loss (normalized MSE)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("PDELab learning curves (1 seed, prototype profile)")
    path = out_dir / "learning_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [str(path)]


def plot_rollout_error(out_dir: Path | None = None) -> list[str]:
    """Per-time error curves for every evaluated run."""
    out_dir = out_dir or (RESULTS / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    found = False
    for d in sorted(RUNS.iterdir()):
        ev = _usable_evaluation(d / "evaluation.json")
        if ev is None:
            continue
        man = _load(d / "manifest.json")
        times = np.asarray(
            man["dataset_manifest"]["system"]["observation_times"][1:]
        )
        ax.plot(times, ev["per_time_nrmse"], label=d.name, lw=1.2)
        found = True
    if not found:
        plt.close(fig)
        return []
    ax.set_xlabel("t")
    ax.set_ylabel("per-snapshot nRMSE")
    ax.set_title("Rollout error vs time (held-out test split)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    path = out_dir / "rollout_error_vs_time.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [str(path)]


def plot_residual_accuracy_tradeoff(out_dir: Path | None = None) -> list[str]:
    """Residual error against data error, with the reference's own value."""
    out_dir = out_dir or (RESULTS / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    found = False
    for d in sorted(RUNS.iterdir()):
        ev = _usable_evaluation(d / "evaluation.json")
        if ev is None:
            continue
        man = _load(d / "manifest.json")
        system = man["experiment"]["system"]
        ax = axes[0] if system == "burgers" else axes[1]
        ax.scatter(
            ev["nrmse"]["mean"], ev["residual_rms_implicit_euler_first_order"],
            s=60, label=d.name,
        )
        ax.axhline(
            ev["reference_residual_rms_implicit_euler_first_order"],
            ls="--", lw=0.8, color="k", alpha=0.5,
        )
        ax.set_xlabel("rollout nRMSE (lower is better)")
        ax.set_ylabel("PDE residual RMS (lower is better)")
        ax.set_title(f"{system}: accuracy vs physics residual")
        ax.grid(alpha=0.3)
        found = True
    if found:
        for ax in axes:
            ax.legend(fontsize=6)
        fig.suptitle(
            "Accuracy / residual trade-off (dashed line = reference solution's "
            "own residual under the identical evaluator, implicit Euler, "
            "interior support)"
        )
    path = out_dir / "residual_accuracy_tradeoff.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [str(path)] if found else []


def plot_throughput_memory(out_dir: Path | None = None) -> list[str]:
    """Throughput and peak memory of each inference variant."""
    out_dir = out_dir or (RESULTS / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    benches = []
    for d in sorted(RUNS.iterdir()):
        b = _load(d / "benchmark_matched.json")
        if b:
            benches.append((d.name, b))
    if not benches:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    variants: list[str] = []
    for _, b in benches:
        for m in b.get("measurements", []):
            if m.get("status") != "failed" and m["variant"] not in variants:
                variants.append(m["variant"])
    for _name, b in benches:
        base = next(
            (m for m in b.get("measurements", [])
             if m["variant"] == "fp32_eager" and m.get("status") != "failed"),
            None,
        )
        if base is None:
            continue
        for m in b.get("measurements", []):
            if m.get("status") == "failed" or m["variant"] == "fp32_eager":
                continue
            axes[0].scatter(
                m["throughput_samples_per_second"],
                m["throughput_samples_per_second"] / base["throughput_samples_per_second"],
                s=70,
            )
            axes[1].scatter(
                m["memory"]["peak_allocated_gib"],
                100.0 * (base["memory"]["peak_allocated_gib"]
                         - m["memory"]["peak_allocated_gib"])
                / base["memory"]["peak_allocated_gib"],
                s=70,
            )
    axes[0].axhline(1.3, ls="--", color="k", lw=0.9)
    axes[0].axhline(1.0, ls=":", color="k", lw=0.9)
    axes[0].set_xlabel("throughput (samples/s)")
    axes[0].set_ylabel("throughput ratio vs FP32 eager")
    axes[0].set_title("inference throughput (matched workload)")
    axes[0].grid(alpha=0.3)
    axes[1].axhline(20.0, ls="--", color="k", lw=0.9)
    axes[1].set_xlabel("peak CUDA allocated (GiB)")
    axes[1].set_ylabel("peak memory reduction (%)")
    axes[1].set_title("inference memory (matched workload)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Inference optimisation: matched checkpoint, grid, horizon, batch size")
    path = out_dir / "throughput_memory.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [str(path)]


def generate_all() -> dict[str, Any]:
    """Generate every figure that the available artifacts support."""
    out_dir = RESULTS / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[str] = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        produced += plot_1d_space_time(d.name, out_dir)
        produced += plot_2d_snapshots(d.name, out_dir)
    produced += plot_learning_curves(out_dir)
    produced += plot_rollout_error(out_dir)
    produced += plot_residual_accuracy_tradeoff(out_dir)
    produced += plot_throughput_memory(out_dir)
    return {
        "status": "completed",
        "n_plots": len(produced),
        "plots": produced,
        "output_dir": str(out_dir),
    }
