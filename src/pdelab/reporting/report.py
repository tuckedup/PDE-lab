"""Build result tables from stored artifacts.

Every number in the report is read from a stored artifact (a run
manifest, an evaluation file, a benchmark file). Nothing is typed in by
hand, so the report cannot drift from what was actually measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import load_yaml, project_root, resolve_path


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != value:  # NaN
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
        evaluation = _read_json(d / "evaluation.json")
        benchmark = _read_json(d / "benchmark_matched.json")
        out.append({
            "dir": d,
            "manifest": manifest,
            "evaluation": evaluation,
            "benchmark": benchmark,
        })
    return out


def build_target_table(runs: list[dict[str, Any]], targets: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate each declared target against the measured artifacts."""
    rows: list[dict[str, Any]] = []
    measured = {}
    for r in runs:
        ev = r["evaluation"]
        if not ev or ev.get("status") != "completed":
            continue
        key = (ev["system"], ev["architecture"], ev["condition"], ev.get("seed"))
        measured[key] = ev

    for target in targets.get("targets", []):
        name = target["name"]
        metric = target["metric"]
        system = target.get("system")
        rows.append({
            "target": name,
            "metric": metric,
            "threshold": target.get("threshold"),
            "comparison": target.get("comparison"),
            "units": target.get("units"),
            "status": "not_run",
            "measured": None,
            "run_id": None,
        })
    return rows


def build_report(output_dir: str | Path = "results") -> dict[str, Any]:
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
            "system": (m.get("experiment") or {}).get("system"),
            "architecture": (m.get("experiment") or {}).get("architecture"),
            "condition": (m.get("experiment") or {}).get("condition"),
            "seed": (m.get("experiment") or {}).get("seed"),
            "lambda_pde": (m.get("experiment") or {}).get("lambda_pde"),
            "status": m.get("status"),
            "best_val_loss": tr.get("best_val_loss"),
            "best_epoch": tr.get("best_epoch"),
            "train_walltime_s": tr.get("walltime_seconds"),
            "params_total": (m.get("parameter_counts") or {}).get("total"),
            "params_complex": (m.get("parameter_counts") or {}).get("complex"),
            "peak_cuda_gib": m.get("peak_cuda_allocated_gib"),
            "rollout_nrmse_mean": (ev or {}).get("nrmse", {}).get("mean"),
            "one_step_nrmse_mean": (ev or {}).get("nrmse_one_step_teacher_forced", {}).get("mean"),
            "persistence_nrmse_mean": (ev or {}).get("persistence_baseline_nrmse", {}).get("mean"),
            "residual_rms": (ev or {}).get("residual_error_physical_rms"),
            "reference_residual_rms": (ev or {}).get("reference_residual_physical_rms"),
            "throughput_ratio": (r["benchmark"] or {}).get("throughput_ratio"),
            "memory_reduction_percent": (r["benchmark"] or {}).get("memory_reduction_percent"),
            "test_dataset_sha256": (ev or {}).get("test_dataset_sha256"),
        })

    # ---- markdown summary ----
    lines: list[str] = []
    lines.append("# PDELab Results\n")
    lines.append(f"Generated from stored artifacts under `runs/`.\n")
    lines.append("## Runs\n")
    if run_rows:
        headers = list(run_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in run_rows:
            lines.append("| " + " | ".join(_fmt(v) for v in row.values()) + " |")
    else:
        lines.append("_No completed runs found._")

    lines.append("\n## Declared targets\n")
    lines.append("Targets are proposals declared in `configs/targets.yaml`; "
                 "they are not accomplishments.\n")
    lines.append("| target | metric | threshold | status | measured | run_id |")
    lines.append("|---|---|---|---|---|---|")
    for t in targets.get("targets", []):
        lines.append(
            f"| {t.get('name')} | {t.get('metric')} | {_fmt(t.get('threshold'))} "
            f"| see status file | - | - |"
        )

    md_path = out_dir / "RESULTS.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                          __import__("time").gmtime()),
                "n_runs": len(run_rows),
                "runs": run_rows,
                "targets": targets.get("targets", []),
            },
            fh,
            indent=2,
            default=str,
        )

    return {
        "status": "completed",
        "n_runs_found": len(run_rows),
        "n_runs_completed": sum(1 for r in run_rows if r["status"] == "completed"),
        "results_markdown": str(md_path),
        "results_json": str(json_path),
    }
