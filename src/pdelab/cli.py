"""PDELab command line interface.

Commands
--------
``pdelab doctor``                     Environment and toolchain check.
``pdelab resources check``            Resource preflight against the profile.
``pdelab data generate``              Generate a dataset for a PDE system.
``pdelab solvers validate``           Numerical validation of a solver.
``pdelab train``                      Train one condition.
``pdelab evaluate``                   Evaluate a trained run on the test split.
``pdelab sweep``                      Plan or execute a run matrix.
``pdelab benchmark``                  Matched inference benchmark.
``pdelab report``                     Build result tables from artifacts.

Execution profiles
------------------
``smoke``      tiny CPU-compatible checks; no long training.
``prototype``  bounded one-seed pilots and local plots.
``full``       the frozen three-seed matrix (must be explicitly requested).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_yaml, project_root, resolve_path
from .resources.preflight import HardwareProfile, JobLock, preflight

DEFAULT_PROFILE = "configs/hardware/atul_laptop.yaml"


def _profile_path(args) -> Path:
    return resolve_path(getattr(args, "hardware", None) or DEFAULT_PROFILE)


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, default=str))


# ----------------------------------------------------------------------
def cmd_doctor(args) -> int:
    """Check the environment, dependencies and project layout."""
    import importlib

    root = project_root()
    report = {
        "project_root": str(root),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {},
    }
    for mod in ["torch", "numpy", "scipy", "h5py", "yaml", "matplotlib", "pytest"]:
        try:
            m = importlib.import_module(mod)
            report["packages"][mod] = getattr(m, "__version__", "present")
        except Exception as exc:
            report["packages"][mod] = f"MISSING ({exc})"

    try:
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if torch.cuda.is_available():
            x = torch.randn(64, 64, device="cuda", requires_grad=True)
            y = (x**2).sum()
            y.backward()
            report["cuda"]["forward_backward_ok"] = bool(
                torch.isfinite(x.grad).all().item()
            )
    except Exception as exc:
        report["cuda"] = {"error": str(exc)}

    report["layout"] = {
        name: (root / name).exists()
        for name in ["configs", "src/pdelab", "tests", "data", "runs", "results", "docs"]
    }
    report["git"] = {}
    from .data.generate import git_revision

    report["git"] = git_revision(root)
    _print_json(report)
    return 0 if all(v != "MISSING" for v in report["packages"].values() if isinstance(v, str) and v.startswith("MISSING")) else 1


def cmd_resources_check(args) -> int:
    """Run the resource preflight."""
    profile = HardwareProfile.load(_profile_path(args))
    report = preflight(profile, require_gpu=not args.cpu_only)
    _print_json(report)
    return 0 if report["ok"] else 2


# ----------------------------------------------------------------------
def cmd_data_generate(args) -> int:
    """Generate a dataset for a PDE system."""
    from .config import PDESystemConfig
    from .data.generate import DatasetGenerator

    cfg_path = resolve_path(args.config)
    raw = load_yaml(cfg_path)
    system = PDESystemConfig.from_dict(raw, source=str(cfg_path))

    profile = args.profile
    sizes = raw.get("profiles", {}).get(profile, {}) or {}
    n_train = int(sizes.get("n_train", args.n_train or 64))
    n_val = int(sizes.get("n_val", args.n_val or 16))
    n_test = int(sizes.get("n_test", args.n_test or 16))
    seed = int(sizes.get("seed", args.seed if args.seed is not None else 0))

    out_dir = resolve_path(args.out or f"data/{system.name}")
    gen = DatasetGenerator(
        system=system,
        out_dir=out_dir,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        seed=seed,
        profile=profile,
    )
    print(f"Generating {system.name} dataset ({profile}): "
          f"train={n_train} val={n_val} test={n_test} seed={seed}")
    manifest = gen.generate(progress=print)
    print(f"Done in {manifest['generation_walltime_seconds']}s -> {manifest['manifest_path']}")
    return 0


def cmd_solvers_validate(args) -> int:
    """Run the numerical validation suite for a solver."""
    from .solvers.validation import run_validation

    result = run_validation(args.system, verbose=True)
    _print_json(result)
    return 0 if result.get("passed") else 1


# ----------------------------------------------------------------------
def cmd_train(args) -> int:
    """Train a single condition."""
    from .training.experiment import ExperimentRunner, load_experiment_config

    cfg_path = resolve_path(args.config)
    cfg, raw = load_experiment_config(cfg_path)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.profile:
        raw.setdefault("profiles", {})
        prof = raw["profiles"].get(args.profile, {}) or {}
        for key in ("epochs", "batch_size", "lr", "max_walltime_minutes"):
            if key in prof:
                setattr(cfg, key, prof[key])
        cfg.tags = [*list(cfg.tags), f"profile:{args.profile}"]

    hw = HardwareProfile.load(_profile_path(args))
    runner = ExperimentRunner(cfg, raw, hw)
    lock = JobLock()
    if not lock.acquire(blocking=False):
        print("ERROR: another PDELab job holds the project lock; refusing to run "
              "concurrently (experiment_concurrency=1).", file=sys.stderr)
        return 3
    try:
        manifest = runner.run()
    finally:
        lock.release()

    status = manifest.get("status")
    print(f"run {runner.run_id}: {status}")
    if status != "completed":
        print(manifest.get("block_reason") or manifest.get("error"), file=sys.stderr)
        return 1
    training = manifest.get("training", {})
    print(f"  best_val_loss={training.get('best_val_loss'):.6g} "
          f"epoch={training.get('best_epoch')} "
          f"walltime={training.get('walltime_seconds', 0):.1f}s")
    return 0


def cmd_evaluate(args) -> int:
    """Evaluate a trained run on the held-out test split."""
    from .evaluation.evaluate import evaluate_run

    run_dir = resolve_path(args.run_id) if os.sep in args.run_id else (
        project_root() / "runs" / args.run_id
    )
    result = evaluate_run(run_dir, device=args.device, split=args.split)
    _print_json({k: v for k, v in result.items() if k != "per_trajectory"})
    return 0 if result.get("status") == "completed" else 1


def cmd_sweep(args) -> int:
    """Plan or execute a run matrix."""
    from .sweep import build_sweep, run_sweep

    cfg_path = resolve_path(args.config)
    plan = build_sweep(cfg_path, args.profile)
    if args.dry_run:
        _print_json(plan)
        return 0
    result = run_sweep(plan, dry_run=False, hardware=_profile_path(args))
    _print_json(result)
    return 0


def cmd_benchmark(args) -> int:
    """Run a matched inference benchmark."""
    from .benchmarks.inference import run_benchmark

    run_dir = project_root() / "runs" / args.run_id
    result = run_benchmark(run_dir, profile=args.profile, device=args.device)
    _print_json(result)
    return 0 if result.get("status") == "completed" else 1


def cmd_report(args) -> int:
    """Build result tables from stored artifacts."""
    from .reporting.report import build_report

    result = build_report(output_dir=resolve_path(args.out))
    _print_json(result)
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdelab",
        description="PDELab - Physics-Constrained Neural Operator research prototype",
    )
    parser.add_argument("--hardware", default=DEFAULT_PROFILE,
                        help="hardware profile YAML (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="environment and toolchain check")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("resources", help="resource management")
    rsub = p.add_subparsers(dest="subcommand", required=True)
    c = rsub.add_parser("check", help="preflight against the hardware profile")
    c.add_argument("--cpu-only", action="store_true",
                   help="do not require a CUDA device")
    c.set_defaults(func=cmd_resources_check)

    p = sub.add_parser("data", help="dataset operations")
    dsub = p.add_subparsers(dest="subcommand", required=True)
    g = dsub.add_parser("generate", help="generate a dataset")
    g.add_argument("--config", required=True)
    g.add_argument("--profile", default="prototype",
                   choices=["smoke", "prototype", "full"])
    g.add_argument("--out", default=None)
    g.add_argument("--n-train", type=int, default=None)
    g.add_argument("--n-val", type=int, default=None)
    g.add_argument("--n-test", type=int, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.set_defaults(func=cmd_data_generate)

    p = sub.add_parser("solvers", help="solver operations")
    ssub = p.add_subparsers(dest="subcommand", required=True)
    v = ssub.add_parser("validate", help="run the numerical validation suite")
    v.add_argument("--system", required=True, choices=["burgers", "allen_cahn"])
    v.set_defaults(func=cmd_solvers_validate)

    p = sub.add_parser("train", help="train one condition")
    p.add_argument("--config", required=True)
    p.add_argument("--profile", default=None, choices=["smoke", "prototype", "full"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a run on the test or validation split")
    p.add_argument("--run-id", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--split", default="test", choices=["test", "val"],
                   help="the test split is the reported held-out metric; the "
                        "validation split is for model selection only")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("sweep", help="plan or run a matrix of experiments")
    p.add_argument("--config", required=True)
    p.add_argument("--profile", default="prototype",
                   choices=["smoke", "prototype", "full"])
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("benchmark", help="matched inference benchmark")
    p.add_argument("--run-id", required=True)
    p.add_argument("--profile", default="matched", choices=["matched", "throughput"])
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("report", help="build result tables from artifacts")
    p.add_argument("--out", default="results")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
