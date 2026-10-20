"""Evaluate a trained run on the held-out test split.

Reports, for the same protocol used for every compared condition:

* free-rollout nRMSE (the primary data metric),
* teacher-forced one-step nRMSE,
* the physical-unit PDE residual error of the rolled-out trajectory,
* the persistence baseline under the identical protocol,
* per-time error curves and per-trajectory values.

The residual error is computed with an identical evaluator, grid, time
spacing and aggregation for every condition. The *reference solution's*
own residual under the same evaluator is reported alongside, so that
operator-specific overfitting is visible.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import project_root
from ..data.dataset import (
    NormalizationStats,
    TrajectoryDataset,
    split_paths,
)
from ..evaluation.metrics import (
    nrmse_per_snapshot,
    nrmse_per_trajectory,
    persistence_forecast,
    residual_rms,
    summarize_nrmse,
)
from ..evaluation.rollout import rollout, teacher_forced_one_step
from ..models.registry import build_model_from_config
from ..physics.operators import (
    allen_cahn_residual,
    burgers_residual,
    d2x_1d,
    laplacian_2d,
    spectral_k_1d,
    spectral_k_2d,
)


def _pair_residuals(
    system: str,
    trajectory: np.ndarray,
    initial: np.ndarray,
    dt: float,
    params: dict[str, float],
    order: int = 1,
) -> float:
    """Physical-unit RMS residual of a trajectory.

    The residual is formed between consecutive states of the trajectory
    itself, with the given initial frame supplying the previous state
    for the first step. For a *model* rollout this measures whether the
    model's own trajectory satisfies the PDE; for the reference it
    measures the reference solution's residual under the identical
    evaluator, grid, time spacing and aggregation.

    ``order`` selects the time-derivative stencil:

    * ``order=1`` (the primary evaluator) uses the forward difference
      ``(u_{i+1} - u_i)/dt``. Its truncation error is O(dt * u_tt), so
      even an exact PDE solution sampled at ``dt`` shows a non-zero
      residual. That floor is reported alongside every measurement.
    * ``order=2`` (a separately declared diagnostic) uses the centred
      second difference ``(u_{i+1} - 2 u_i + u_{i-1})/dt^2``, which
      cancels the O(dt) term. It is reported with a *distinct* name and
      is never mixed with the primary evaluator, so that a result that
      is specific to one operator can be exposed.
    """
    traj = np.asarray(trajectory, dtype=np.float32)
    init = np.asarray(initial, dtype=np.float32)
    if traj.ndim == init.ndim + 1:
        prev = np.concatenate([init[:, None, ...], traj[:, :-1, ...]], axis=1)
    else:
        raise ValueError("trajectory must have one more axis than initial")
    cur = torch.from_numpy(traj)
    prev_t = torch.from_numpy(prev)

    if order == 1:
        if system == "burgers":
            r = burgers_residual(cur, prev_t, dt=dt, nu=params.get("nu", 0.01))
        elif system == "allen_cahn":
            r = allen_cahn_residual(
                cur, prev_t, dt=dt, D=params.get("D", 0.001), a=params.get("a", 1.0)
            )
        else:
            raise ValueError(f"unknown system {system}")
        return residual_rms(r)

    if order == 2:
        if traj.shape[1] < 3:
            return float("nan")
        # Centred second-order time difference on the interior snapshots.
        centre = cur[:, 1:-1]
        nxt = cur[:, 2:]
        prv = cur[:, :-2]
        u_t = (nxt - 2.0 * centre + prv) / (dt * dt)
        if system == "burgers":
            nu = params.get("nu", 0.01)
            k = spectral_k_1d(centre.shape[-1], device=centre.device, dtype=centre.dtype)
            flux_hat = torch.fft.fft(0.5 * centre**2, dim=-1)
            flux_x = torch.fft.ifft(1j * k * flux_hat, dim=-1).real
            u_xx = d2x_1d(centre, k)
            r = u_t + flux_x - nu * u_xx
        elif system == "allen_cahn":
            D = params.get("D", 0.001)
            a = params.get("a", 1.0)
            kx, ky = spectral_k_2d(
                centre.shape[-2], centre.shape[-1],
                device=centre.device, dtype=centre.dtype,
            )
            lap = laplacian_2d(centre, kx, ky)
            r = u_t - D * lap - a * (centre - centre**3)
        else:
            raise ValueError(f"unknown system {system}")
        return residual_rms(r)

    raise ValueError(f"unknown residual order {order!r}")


def evaluate_run(run_dir: str | Path, device: str = "auto") -> dict[str, Any]:
    """Evaluate a trained run and write ``evaluation.json`` next to it."""
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = project_root() / run_dir
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {"status": "failed", "error": f"no manifest at {manifest_path}"}
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    if manifest.get("status") != "completed":
        return {
            "status": "skipped",
            "reason": f"run status is {manifest.get('status')!r}",
            "run_id": manifest.get("run_id"),
        }

    cfg = manifest["experiment"]
    system = cfg["system"]
    dataset_dir = cfg["dataset_dir"]
    paths = split_paths(dataset_dir, system)

    dev = torch.device(
        "cuda" if (device == "auto" and torch.cuda.is_available()) else
        ("cpu" if device == "auto" else device)
    )

    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        return {"status": "failed", "error": f"no checkpoint at {ckpt_path}"}
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)

    model = build_model_from_config(
        {
            "architecture": cfg["architecture"],
            "history": cfg.get("history", 1),
            **{k: v for k, v in (cfg.get("model_kwargs") or {}).items()},
        },
        dimension=2 if system == "allen_cahn" else 1,
    )
    model.load_state_dict(ckpt["model"])
    model.to(dev)
    model.eval()

    stats = NormalizationStats.from_dict(ckpt["normalization"])
    history_len = int(cfg.get("history", 1))
    test_set = TrajectoryDataset(paths["test"], stats, history=history_len, stride=1)

    obs_times = manifest["dataset_manifest"]["system"]["observation_times"]
    dt = float(obs_times[1] - obs_times[0])
    params = manifest["dataset_manifest"]["system"]["params"]

    started = time.time()
    n_steps = len(obs_times) - history_len

    # ---- free rollout (primary protocol) ----
    spatial = test_set.u.shape[2:]
    preds = np.zeros((test_set.n_traj, n_steps, *spatial), dtype=np.float32)
    hist_arr = np.zeros((test_set.n_traj, history_len, *spatial), dtype=np.float32)
    for t in range(test_set.n_traj):
        h = torch.from_numpy(test_set.u[t, :history_len]).unsqueeze(0)
        out = rollout(model, h, n_steps, stats, params=None, device=dev)
        # The model emits (B, 1, *spatial); drop the singleton channel.
        preds[t] = out[0, :, 0].cpu().numpy()
        hist_arr[t] = test_set.u[t, :history_len]

    reference = test_set.u[:, history_len : history_len + n_steps].astype(np.float32)

    per_traj = nrmse_per_trajectory(preds, reference)
    energy = np.sum(reference**2, axis=tuple(range(1, reference.ndim)))
    summary = summarize_nrmse(per_traj, energy)
    per_snap = nrmse_per_snapshot(preds, reference)

    # ---- persistence baseline under the identical protocol ----
    # The forecast repeats the *initial* frame along the time axis. The
    # helper is used rather than an inline np.repeat so that the axis
    # being tiled is always the time axis.
    persistence = persistence_forecast(hist_arr[:, -1], n_steps).astype(np.float32)
    persistence_per_traj = nrmse_per_trajectory(persistence, reference)
    persistence_summary = summarize_nrmse(persistence_per_traj, energy)

    # ---- teacher-forced one-step ----
    tf_pred, tf_ref = teacher_forced_one_step(model, test_set, stats, device=dev)
    tf_per_traj = nrmse_per_trajectory(tf_pred, tf_ref)
    tf_summary = summarize_nrmse(tf_per_traj)

    # ---- physics residual error ----
    initial_frame = hist_arr[:, -1]
    resid = _pair_residuals(system, preds, initial_frame, dt, params, order=1)
    ref_resid = _pair_residuals(system, reference, initial_frame, dt, params, order=1)
    # Separately declared diagnostic evaluator (centred second-order time
    # difference). Reported under a distinct name; never mixed with the
    # primary evaluator.
    resid_o2 = _pair_residuals(system, preds, initial_frame, dt, params, order=2)
    ref_resid_o2 = _pair_residuals(
        system, reference, initial_frame, dt, params, order=2
    )

    result = {
        "status": "completed",
        "run_id": manifest["run_id"],
        "system": system,
        "architecture": cfg["architecture"],
        "condition": cfg["condition"],
        "seed": cfg.get("seed"),
        "lambda_pde": cfg.get("lambda_pde", 0.0),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "test_dataset_sha256": manifest["dataset_manifest"]["files"]["test"]["sha256"],
        "protocol": {
            "type": "free_rollout",
            "observation_dt": dt,
            "n_rollout_steps": n_steps,
            "n_test_trajectories": int(test_set.n_traj),
            "history_length": history_len,
            "copied_initial_frame_included_in_error": False,
            "epsilon": 1e-12,
        },
        "nrmse": summary.to_dict(),
        "nrmse_one_step_teacher_forced": tf_summary.to_dict(),
        "persistence_baseline_nrmse": persistence_summary.to_dict(),
        "residual_error_physical_rms": resid,
        "reference_residual_physical_rms": ref_resid,
        "residual_error_physical_rms_order2": resid_o2,
        "reference_residual_physical_rms_order2": ref_resid_o2,
        "per_time_nrmse": per_snap.mean(axis=0).tolist(),
        "per_trajectory_nrmse": per_traj.tolist(),
        "evaluation_walltime_seconds": round(time.time() - started, 3),
    }
    out_path = run_dir / "evaluation.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    result["evaluation_path"] = str(out_path)
    return result
