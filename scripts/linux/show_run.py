"""Inspect a run: manifest, evaluation and benchmark artifacts.

Usage:
    python scripts/linux/show_run.py <run_id> [--full]
"""

import json
import sys
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run_id = sys.argv[1]
    full = "--full" in sys.argv
    run_dir = ROOT / "runs" / run_id
    if not run_dir.exists():
        print(f"no such run: {run_dir}")
        return 1

    manifest = json.loads((run_dir / "manifest.json").read_text())
    exp = manifest.get("experiment", {})
    training = manifest.get("training", {})

    print(f"run_id            : {manifest.get('run_id')}")
    print(f"status            : {manifest.get('status')}")
    print(f"created_utc       : {manifest.get('created_utc')}")
    print(f"system/arch/cond : {exp.get('system')} / {exp.get('architecture')} / "
          f"{exp.get('condition')}")
    print(f"seed              : {exp.get('seed')}")
    print(f"lambda_pde        : {exp.get('lambda_pde')}")
    print(f"best_val_loss     : {training.get('best_val_loss')}")
    print(f"best_epoch        : {training.get('best_epoch')}")
    print(f"train_walltime_s  : {training.get('walltime_seconds')}")
    print(f"stopped_reason    : {training.get('stopped_reason')}")
    print(f"param_counts      : {manifest.get('parameter_counts')}")
    print(f"peak_cuda_gib     : {manifest.get('peak_cuda_allocated_gib')}")
    print(f"peak_host_rss_gib : {manifest.get('peak_host_rss_gib')}")
    print(f"git               : {manifest.get('git')}")
    print(f"dataset_test_sha  : "
          f"{(manifest.get('dataset_manifest') or {}).get('files', {}).get('test', {}).get('sha256')}")
    pg = training.get("physics_grad_check")
    if pg:
        print(f"physics_grad_check: {json.dumps(pg)}")
    else:
        print("physics_grad_check: n/a (data-only condition)")

    ev_path = run_dir / "evaluation.json"
    if ev_path.exists():
        ev = json.loads(ev_path.read_text())
        print("\n--- evaluation ---")
        print(f"protocol          : {json.dumps(ev.get('protocol'))}")
        print(f"rollout nRMSE     : mean={ev['nrmse']['mean']:.6g} "
              f"median={ev['nrmse']['median']:.6g} std={ev['nrmse']['std']:.6g}")
        print(f"one-step nRMSE    : mean={ev['nrmse_one_step_teacher_forced']['mean']:.6g}")
        print(f"persistence nRMSE : mean={ev['persistence_baseline_nrmse']['mean']:.6g}")
        print(f"residual rms      : {ev.get('residual_error_physical_rms'):.6g}")
        print(f"reference residual: {ev.get('reference_residual_physical_rms'):.6g}")
        if ev.get("residual_error_physical_rms_order2") is not None:
            print(f"residual rms (o2) : {ev.get('residual_error_physical_rms_order2'):.6g}"
                  f"   (reference {ev.get('reference_residual_physical_rms_order2'):.6g})")
        if full:
            print(f"per_time_nrmse    : "
                  f"{[round(v, 5) for v in ev.get('per_time_nrmse', [])]}")

    bench_path = run_dir / "benchmark_matched.json"
    if bench_path.exists():
        b = json.loads(bench_path.read_text())
        print("\n--- benchmark (matched) ---")
        print(f"workload          : {json.dumps(b.get('workload'))}")
        for name, v in b.get("variants", {}).items():
            t = v.get("timing", {})
            m = v.get("memory", {})
            print(f"  {name}: median={t.get('median_seconds', float('nan')):.6g}s "
                  f"peak_alloc={m.get('peak_allocated_gib', float('nan')):.4f} GiB "
                  f"acc_ok={v.get('accuracy_within_tolerance')}")
        print(f"throughput_ratio  : {b.get('throughput_ratio')}")
        print(f"memory_reduction  : {b.get('memory_reduction_percent')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
