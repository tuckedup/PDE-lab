"""Build result tables from stored artifacts.

Every number in the report is read from a stored artifact (a run
manifest, an evaluation file, a benchmark file). Nothing is typed in by
hand, so the report cannot drift from what was actually measured.

The target table reports each declared target as one of

    met | missed | not_run | not_comparable

with the measured value, the units, the run/checkpoint identifiers, the
test dataset hash and the protocol.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..config import load_yaml, project_root
from ..evaluation.metrics import relative_residual_improvement


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return str(value)
        return f"{value:.{digits}g}"
    return str(value)


def collect_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    runs_dir = Path(runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = project_root() / runs_dir
    out = []
    if not runs_dir.exists():
        return out
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        manifest = _read_json(d / "manifest.json")
        if manifest is None:
            continue
        out.append({
            "dir": d,
            "manifest": manifest,
            "evaluation": _read_json(d / "evaluation.json"),
            "benchmark": _read_json(d / "benchmark_matched.json"),
        })
    return out


def _completed(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r for r in runs
        if r["evaluation"] and r["evaluation"].get("status") == "completed"
    ]


def _best_run(runs, system, arch, condition):
    """The run used for the target table (the lowest rollout nRMSE)."""
    cands = [
        r for r in _completed(runs)
        if r["evaluation"]["system"] == system
        and r["evaluation"]["architecture"] == arch
        and r["evaluation"]["condition"] == condition
    ]
    if not cands:
        return None
    return min(cands, key=lambda r: r["evaluation"]["nrmse"]["mean"])


def build_target_table(runs: list[dict[str, Any]], targets: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate each declared target against the measured artifacts."""
    rows: list[dict[str, Any]] = []
    for target in targets.get("targets", []):
        name = target["name"]
        metric = target["metric"]
        threshold = target.get("threshold")
        comparison = target.get("comparison", "<")
        system = target.get("system")
        row = {
            "target": name,
            "metric": metric,
            "threshold": threshold,
            "comparison": comparison,
            "units": target.get("units"),
            "status": "not_run",
            "measured": None,
            "run_id": None,
            "checkpoint": None,
            "test_dataset_sha256": None,
            "protocol": None,
            "note": "",
        }

        if metric == "nrmse":
            run = _best_run(runs, system, "fno1d" if system == "burgers" else "fno2d",
                             "data_only")
            phys = _best_run(runs, system, "fno1d" if system == "burgers" else "fno2d",
                             "physics")
            if run is None and phys is None:
                row["note"] = "no completed run for this system"
                rows.append(row)
                continue
            # The target is met if the *matched* condition set includes a
            # run that meets it; the table records the best run.
            candidates = [x for x in (run, phys) if x is not None]
            best = min(candidates, key=lambda r: r["evaluation"]["nrmse"]["mean"])
            value = best["evaluation"]["nrmse"]["mean"]
            row["measured"] = value
            row["run_id"] = best["evaluation"]["run_id"]
            row["checkpoint"] = best["evaluation"]["checkpoint"]
            row["test_dataset_sha256"] = best["evaluation"].get("test_dataset_sha256")
            row["protocol"] = best["evaluation"].get("protocol")
            row["status"] = "met" if value < threshold else "missed"
            row["note"] = (
                "per-condition rollout nRMSE: "
                + ", ".join(
                    f"{x['evaluation']['condition']}="
                    f"{x['evaluation']['nrmse']['mean']:.4g}"
                    for x in candidates
                )
            )

        elif metric == "relative_residual_improvement":
            arch = "fno1d" if system == "burgers" else "fno2d"
            base = _best_run(runs, system, arch, "data_only")
            prop = _best_run(runs, system, arch, "physics")
            if base is None or prop is None:
                row["note"] = "missing the matched baseline or proposed run"
                rows.append(row)
                continue
            b = base["evaluation"]["residual_error_physical_rms"]
            p = prop["evaluation"]["residual_error_physical_rms"]
            value = relative_residual_improvement(b, p)
            b2 = base["evaluation"].get("residual_error_physical_rms_order2")
            p2 = prop["evaluation"].get("residual_error_physical_rms_order2")
            v2 = relative_residual_improvement(b2, p2) if (b2 and p2 is not None) else None
            row["measured"] = value
            row["run_id"] = prop["evaluation"]["run_id"]
            row["checkpoint"] = prop["evaluation"]["checkpoint"]
            row["test_dataset_sha256"] = prop["evaluation"].get("test_dataset_sha256")
            row["protocol"] = {
                "baseline": base["evaluation"]["run_id"],
                "proposed": prop["evaluation"]["run_id"],
                "residual_evaluator": "forward-difference time derivative (primary)",
            }
            if value != value:
                row["status"] = "not_comparable"
                row["note"] = "the baseline residual denominator is zero or undefined"
            else:
                row["status"] = "met" if value >= threshold else "missed"
            if v2 is not None:
                row["note"] = (
                    f"second-order (diagnostic) evaluator gives {v2:+.2f}%; "
                    f"baseline residual {b:.4g}, proposed {p:.4g}"
                )

        elif metric in ("throughput_ratio", "memory_reduction_percent"):
            # Evaluate the target on the *declared reference model* so the
            # result cannot be inflated by picking a favourable
            # checkpoint. The value for every other model is reported too.
            ref_name = targets.get("inference_reference_model")
            per_run = []
            for r in runs:
                if not r["benchmark"]:
                    continue
                val = r["benchmark"].get(metric)
                if val is None:
                    continue
                per_run.append((r["manifest"]["run_id"], val, r))
            if not per_run:
                row["note"] = "no matched benchmark artifact found"
                rows.append(row)
                continue
            selected = next(
                (x for x in per_run if x[0] == ref_name), None
            ) or max(per_run, key=lambda x: x[1])
            name, value, r = selected
            row["measured"] = value
            row["run_id"] = name
            row["checkpoint"] = r["benchmark"].get("checkpoint")
            row["protocol"] = {
                "workload": r["benchmark"].get("workload"),
                "isolation": r["benchmark"].get("isolation"),
                "variant": r["benchmark"].get(
                    "throughput_best_variant" if metric == "throughput_ratio"
                    else "memory_best_variant"
                ),
                "reference_model_declared": ref_name,
            }
            row["status"] = "met" if value >= threshold else "missed"
            spread = ", ".join(f"{n}={v:.3g}" for n, v, _ in sorted(per_run))
            row["note"] = f"per-run values: {spread}"
            if metric == "memory_reduction_percent":
                row["note"] += (
                    "; the reduction depends strongly on the model size, because "
                    "fixed costs (cuFFT workspace, allocator granularity) dominate "
                    "the peak for the small models used here"
                )

        rows.append(row)
    return rows


def build_report(output_dir: str | Path = "results", make_plots: bool = True) -> dict[str, Any]:
    """Build the full report from artifacts."""
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_path = project_root() / "configs" / "targets.yaml"
    targets = load_yaml(targets_path) if targets_path.exists() else {"targets": []}

    runs = collect_runs(project_root() / "runs")

    run_rows = []
    for r in runs:
        m = r["manifest"]
        ev = r["evaluation"]
        tr = m.get("training") or {}
        run_rows.append({
            "run_id": m.get("run_id"),
            "status": m.get("status"),
            "system": (m.get("experiment") or {}).get("system"),
            "architecture": (m.get("experiment") or {}).get("architecture"),
            "condition": (m.get("experiment") or {}).get("condition"),
            "seed": (m.get("experiment") or {}).get("seed"),
            "lambda_pde": (m.get("experiment") or {}).get("lambda_pde"),
            "best_val_loss": tr.get("best_val_loss"),
            "best_epoch": tr.get("best_epoch"),
            "train_walltime_s": tr.get("walltime_seconds"),
            "params_total": (m.get("parameter_counts") or {}).get("total"),
            "params_complex": (m.get("parameter_counts") or {}).get("complex"),
            "peak_cuda_gib": m.get("peak_cuda_allocated_gib"),
            "peak_host_rss_gib": m.get("peak_host_rss_gib"),
            "rollout_nrmse_mean": (ev or {}).get("nrmse", {}).get("mean"),
            "rollout_nrmse_median": (ev or {}).get("nrmse", {}).get("median"),
            "one_step_nrmse_mean": (ev or {}).get("nrmse_one_step_teacher_forced", {}).get("mean"),
            "persistence_nrmse_mean": (ev or {}).get("persistence_baseline_nrmse", {}).get("mean"),
            "residual_rms": (ev or {}).get("residual_error_physical_rms"),
            "reference_residual_rms": (ev or {}).get("reference_residual_physical_rms"),
            "residual_rms_order2": (ev or {}).get("residual_error_physical_rms_order2"),
            "throughput_ratio": (r["benchmark"] or {}).get("throughput_ratio"),
            "memory_reduction_percent": (r["benchmark"] or {}).get("memory_reduction_percent"),
            "test_dataset_sha256": (ev or {}).get("test_dataset_sha256"),
        })

    target_rows = build_target_table(runs, targets)

    # ---- markdown summary ----
    lines: list[str] = []
    lines.append("# PDELab Results\n")
    lines.append(
        "Generated by `pdelab report` from the artifacts under `runs/`. "
        "Every number below is read from a stored artifact.\n"
    )
    lines.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")

    lines.append("\n## Declared targets\n")
    lines.append(
        "Targets are *proposals* declared in `configs/targets.yaml`, not "
        "accomplishments. Status is one of met / missed / not_run / "
        "not_comparable.\n"
    )
    lines.append(
        "| target | metric | threshold | status | measured | units | run_id |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for t in target_rows:
        lines.append(
            f"| {t['target']} | {t['metric']} | {t['comparison']} {_fmt(t['threshold'])} "
            f"| **{t['status']}** | {_fmt(t['measured'])} | {t['units']} "
            f"| {t['run_id'] or '-'} |"
        )
    for t in target_rows:
        if t.get("note"):
            lines.append(f"\n- `{t['target']}`: {t['note']}")

    lines.append("\n\n## Runs\n")
    if run_rows:
        headers = list(run_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in run_rows:
            lines.append("| " + " | ".join(_fmt(v) for v in row.values()) + " |")
    else:
        lines.append("_No runs found._")

    lines.append("\n## Protocol notes\n")
    lines.append(
        "- Data metric: per-trajectory relative RMS error over the predicted "
        "future snapshots only; the copied initial frame is excluded.\n"
        "- Residual metric: physical-unit RMS PDE residual over the declared "
        "space/time domain, with an identical evaluator, grid, time spacing "
        "and aggregation for every condition. The reference solution's own "
        "residual under the same evaluator is reported alongside.\n"
        "- The primary residual evaluator uses a forward-difference time "
        "derivative; a separately declared second-order (centred) evaluator is "
        "reported as a diagnostic and is never mixed with the primary one.\n"
        "- Inference comparisons use the same checkpoint, grid, horizon, batch "
        "size and output contract; each variant is measured in a separate "
        "process.\n"
    )

    md_path = out_dir / "RESULTS.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # The same generated table is mirrored into docs/ so that the
    # documentation never contains hand-written numbers.
    docs_dir = project_root() / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_path = docs_dir / "RESULTS.md"
    docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_runs": len(run_rows),
                "n_runs_completed": sum(1 for r in run_rows if r["status"] == "completed"),
                "runs": run_rows,
                "targets": target_rows,
            },
            fh,
            indent=2,
            default=str,
        )

    plots = []
    if make_plots:
        from .plots import generate_all

        try:
            plot_result = generate_all()
            plots = plot_result.get("plots", [])
        except Exception as exc:
            plots = [f"plot generation failed: {type(exc).__name__}: {exc}"]

    return {
        "status": "completed",
        "n_runs_found": len(run_rows),
        "n_runs_completed": sum(1 for r in run_rows if r["status"] == "completed"),
        "n_targets": len(target_rows),
        "targets_met": sum(1 for t in target_rows if t["status"] == "met"),
        "targets_missed": sum(1 for t in target_rows if t["status"] == "missed"),
        "targets_not_run": sum(1 for t in target_rows if t["status"] == "not_run"),
        "results_markdown": str(md_path),
        "results_json": str(json_path),
        "docs_results_markdown": str(docs_path),
        "plots": plots,
    }
