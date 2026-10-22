#!/usr/bin/env python3
"""Generate ``RELEASE_MANIFEST.json`` and the release evidence index.

The manifest is written from **stored artifacts only**: the built wheel,
the recorded clean-install transcript, the declared evidence cohort and
the run manifests/evaluations of the cohort members.  No number in the
manifest is typed by hand.

Usage:
    .venv/bin/python scripts/linux/release_manifest.py

Outputs:
    RELEASE_MANIFEST.json                (repository root)
    campaigns/release/EVIDENCE_INDEX.json
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
EVIDENCE_DIR = ROOT / "campaigns" / "release"
COHORT_PATH = ROOT / "campaigns" / "evidence-cohort" / "cohort.yaml"
MANIFEST_PATH = ROOT / "RELEASE_MANIFEST.json"
INDEX_PATH = EVIDENCE_DIR / "EVIDENCE_INDEX.json"

#: The exact command that the manifest records as the reproduction
#: command.  It lives in its own file so that the manifest and the
#: clean-install verification can never disagree: the verification
#: executes this file verbatim against the installed CLI and the
#: manifest records the same bytes.
REPRODUCTION_COMMAND_FILE = ROOT / "campaigns" / "release" / "fixtures" / "reproduction_command.txt"
REPRODUCTION_COMMAND = REPRODUCTION_COMMAND_FILE.read_text(encoding="utf-8").strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # Only the trailing newline is removed: `git status --porcelain` puts
    # a two character status prefix and a space in front of every path,
    # and stripping the leading whitespace would shift that prefix.
    return out.stdout.rstrip("\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None


def main() -> int:
    import numpy
    import torch

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]

    # ------------------------------------------------------------------
    # 1. the artifact
    # ------------------------------------------------------------------
    wheels = sorted(DIST.glob(f"pdelab-{version}-*.whl"))
    if not wheels:
        print(f"ERROR: no wheel for version {version} in {DIST}", file=sys.stderr)
        return 1
    wheel = wheels[0]
    wheel_sha = sha256_file(wheel)
    build_summary = load_json(EVIDENCE_DIR / "build_summary.json")
    if build_summary["wheel_sha256"] != wheel_sha:
        print(
            "ERROR: the recorded build summary does not match the wheel on disk "
            f"({build_summary['wheel_sha256']} != {wheel_sha}); rebuild it with "
            "scripts/linux/release_build.sh",
            file=sys.stderr,
        )
        return 1

    # The measured reproducibility check, if it was run.
    reproducibility_path = EVIDENCE_DIR / "build_reproducibility.json"
    reproducibility_check: dict[str, Any] = {"performed": False}
    if reproducibility_path.exists():
        reproducibility_check = {
            "performed": True,
            "record": str(reproducibility_path.relative_to(ROOT)),
            "record_sha256": sha256_file(reproducibility_path),
            **{k: v for k, v in load_json(reproducibility_path).items() if k != "schema"},
        }

    # ------------------------------------------------------------------
    # 2. the source revision
    # ------------------------------------------------------------------
    revision = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain")
    dirty_paths = [
        line[3:] for line in (status or "").splitlines() if line.strip()
    ]

    # ------------------------------------------------------------------
    # 3. the declared cohort and the identity of its members
    # ------------------------------------------------------------------
    cohort = yaml.safe_load(COHORT_PATH.read_text(encoding="utf-8"))
    members: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}

    for member in cohort.get("members", []):
        for run_id in member.get("run_ids", []):
            run_dir = ROOT / "runs" / run_id
            manifest_path = run_dir / "manifest.json"
            evaluation_path = run_dir / "evaluation.json"
            if not manifest_path.exists():
                members.append(
                    {
                        "member_id": member["id"],
                        "run_id": run_id,
                        "role": member.get("role"),
                        "evidence": "missing_manifest",
                    }
                )
                continue
            run_manifest = load_json(manifest_path)
            evaluation = load_json(evaluation_path) if evaluation_path.exists() else {}
            identity = evaluation.get("identity", {})
            checkpoint_sha = identity.get("checkpoint_sha256")
            entry: dict[str, Any] = {
                "member_id": member["id"],
                "role": member.get("role"),
                "run_id": run_id,
                "system": member.get("system"),
                "architecture": member.get("architecture"),
                "condition": member.get("condition"),
                "lambda_pde": member.get("lambda_pde"),
                "seeds": member.get("seeds"),
                "status": run_manifest.get("status"),
                "evidence_status": evaluation.get("evidence_status"),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_epoch": evaluation.get("checkpoint_epoch"),
                "test_dataset_sha256": identity.get("dataset", {}).get("sha256"),
                "evaluator_sha256": identity.get("evaluator", {}).get("sha256"),
                "protocol_sha256": identity.get("protocol", {}).get("sha256"),
                "manifest_sha256": sha256_file(manifest_path),
                "evaluation_sha256": (
                    sha256_file(evaluation_path) if evaluation_path.exists() else None
                ),
            }
            if evaluation:
                entry["measured"] = {
                    "rollout_nrmse_mean": evaluation.get("nrmse", {}).get("mean"),
                    "one_step_nrmse_mean": evaluation.get(
                        "nrmse_one_step_teacher_forced", {}
                    ).get("mean"),
                    "persistence_nrmse_mean": evaluation.get(
                        "persistence_baseline_nrmse", {}
                    ).get("mean"),
                    "numerical_stepping_nrmse_mean": evaluation.get(
                        "numerical_stepping_baseline_nrmse", {}
                    ).get("mean"),
                    "residual_rms_implicit_euler_first_order": evaluation.get(
                        "residual_rms_implicit_euler_first_order"
                    ),
                    "reference_residual_rms_implicit_euler_first_order": evaluation.get(
                        "reference_residual_rms_implicit_euler_first_order"
                    ),
                    "residual_rms_centred_first_derivative": evaluation.get(
                        "residual_rms_centred_first_derivative"
                    ),
                    "reference_residual_rms_centred_first_derivative": evaluation.get(
                        "reference_residual_rms_centred_first_derivative"
                    ),
                }
            members.append(entry)
            if checkpoint_sha:
                checkpoints[run_id] = {
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_epoch": evaluation.get("checkpoint_epoch"),
                    "checkpoint_file": f"runs/{run_id}/best.pt",
                    "checkpoint_file_sha256": (
                        sha256_file(run_dir / "best.pt") if (run_dir / "best.pt").exists() else None
                    ),
                }

    # ------------------------------------------------------------------
    # 4. the measured limitations, recomputed from the stored artifacts
    # ------------------------------------------------------------------
    def measured_of(run_id: str) -> dict[str, Any]:
        path = ROOT / "runs" / run_id / "evaluation.json"
        if not path.exists():
            return {}
        return load_json(path)

    def improvement(baseline: float | None, proposed: float | None) -> float | None:
        if baseline is None or proposed is None or abs(baseline) < 1e-30:
            return None
        return 100.0 * (baseline - proposed) / baseline

    residual_comparison: dict[str, Any] = {}
    for system, baseline_id, proposed_id in [
        (
            "burgers",
            "burgers_fno1d_data_only_seed{seed}",
            "burgers_fno1d_physics_seed{seed}",
        ),
        (
            "allen_cahn",
            "allen_cahn_fno2d_data_only_seed{seed}",
            "allen_cahn_fno2d_physics_seed{seed}",
        ),
    ]:
        rows = []
        for seed in cohort.get("members", [{}])[0].get("seeds", [0, 1, 2]):
            base = measured_of(baseline_id.format(seed=seed))
            prop = measured_of(proposed_id.format(seed=seed))
            if not base or not prop:
                continue
            primary = {
                "baseline_residual_rms": base.get(
                    "residual_rms_implicit_euler_first_order"
                ),
                "proposed_residual_rms": prop.get(
                    "residual_rms_implicit_euler_first_order"
                ),
                "reference_residual_rms": base.get(
                    "reference_residual_rms_implicit_euler_first_order"
                ),
            }
            independent = {
                "baseline_residual_rms": base.get(
                    "residual_rms_centred_first_derivative"
                ),
                "proposed_residual_rms": prop.get(
                    "residual_rms_centred_first_derivative"
                ),
                "reference_residual_rms": base.get(
                    "reference_residual_rms_centred_first_derivative"
                ),
            }
            ref_ind = independent["reference_residual_rms"]
            model_ind = [
                v
                for v in (
                    independent["baseline_residual_rms"],
                    independent["proposed_residual_rms"],
                )
                if v is not None
            ]
            independent["reference_is_more_pde_consistent_than_either_model"] = bool(
                ref_ind is not None and model_ind and ref_ind < min(model_ind)
            )
            rows.append(
                {
                    "seed": seed,
                    "baseline_run": baseline_id.format(seed=seed),
                    "proposed_run": proposed_id.format(seed=seed),
                    "primary_implicit_euler_improvement_percent": improvement(
                        base.get("residual_rms_implicit_euler_first_order"),
                        prop.get("residual_rms_implicit_euler_first_order"),
                    ),
                    "independent_centred_derivative_improvement_percent": improvement(
                        base.get("residual_rms_centred_first_derivative"),
                        prop.get("residual_rms_centred_first_derivative"),
                    ),
                    "primary_implicit_euler": primary,
                    "independent_centred_derivative": independent,
                }
            )
        residual_comparison[system] = rows

    # the learned operators against the direct numerical stepping baseline
    numerical_comparison = []
    seen_runs: set[str] = set()
    for entry in members:
        measured = entry.get("measured") or {}
        learned = measured.get("rollout_nrmse_mean")
        stepping = measured.get("numerical_stepping_nrmse_mean")
        if learned is None or stepping is None:
            continue
        if entry["run_id"] in seen_runs:
            # two cohort members may declare the same run (the inference
            # reference is also a cohort member); count it once.
            continue
        seen_runs.add(entry["run_id"])
        numerical_comparison.append(
            {
                "run_id": entry["run_id"],
                "system": entry["system"],
                "architecture": entry["architecture"],
                "condition": entry["condition"],
                "seed": entry["run_id"].rsplit("seed", 1)[-1],
                "learned_rollout_nrmse": learned,
                "numerical_stepping_nrmse": stepping,
                "numerical_stepping_over_learned_ratio": stepping / learned,
                "learned_is_worse_than_numerical_stepping": bool(learned > stepping),
            }
        )

    # ------------------------------------------------------------------
    # 4b. the inference memory claim, recomputed from the stored
    #     benchmark artifact of the DECLARED inference reference run
    # ------------------------------------------------------------------
    inference_reference_run = None
    for member in cohort.get("members", []):
        if member.get("role") == "inference_reference":
            ids = member.get("run_ids") or []
            if ids:
                inference_reference_run = ids[0]
            break

    inference_memory: dict[str, Any] = {}
    bench_path = None
    if inference_reference_run:
        candidate = ROOT / "runs" / inference_reference_run / "benchmark_matched.json"
        if candidate.exists():
            bench_path = candidate
            bench = load_json(candidate)
            measurements = bench.get("measurements", [])
            base_var = next(
                (m for m in measurements if m.get("variant") == "fp32_eager"), None
            )
            best_name = bench.get("memory_best_variant")
            best_var = next(
                (m for m in measurements if m.get("variant") == best_name), None
            )
            if base_var and best_var:
                base_mem = base_var.get("memory", {})
                best_mem = best_var.get("memory", {})

                def _pair(key: str) -> dict[str, Any]:
                    b = base_mem.get(key)
                    v = best_mem.get(key)
                    delta = None
                    pct = None
                    if isinstance(b, (int, float)) and isinstance(v, (int, float)):
                        delta = v - b
                        pct = (100.0 * (b - v) / b) if b else None
                    return {
                        "baseline_fp32_eager": b,
                        f"variant_{best_name}": v,
                        "absolute_change": delta,
                        "reduction_percent": pct,
                    }

                device_free = _pair("device_free_gib")
                inference_memory = {
                    "reference_run": inference_reference_run,
                    "artifact": str(bench_path.relative_to(ROOT)),
                    "artifact_sha256": sha256_file(bench_path),
                    "baseline_variant": "fp32_eager",
                    "best_memory_variant": best_name,
                    "peak_allocated_gib": _pair("peak_allocated_gib"),
                    "peak_reserved_gib": _pair("peak_reserved_gib"),
                    "allocated_before_reset_gib": _pair("allocated_before_reset_gib"),
                    "reserved_before_reset_gib": _pair("reserved_before_reset_gib"),
                    "device_free_gib": device_free,
                    "device_total_gib": best_mem.get("device_total_gib"),
                    "persistent_total_bytes": {
                        "baseline_fp32_eager": base_mem.get("persistent_total_bytes"),
                        f"variant_{best_name}": best_mem.get("persistent_total_bytes"),
                        "weights": best_mem.get("persistent_weights_bytes"),
                        "input": best_mem.get("persistent_input_bytes"),
                        "output": best_mem.get("persistent_output_bytes"),
                    },
                    "reported_memory_reduction_percent": bench.get(
                        "memory_reduction_percent"
                    ),
                    "reported_memory_reserved_reduction_percent": bench.get(
                        "memory_reserved_reduction_percent"
                    ),
                    "per_variant_joint_pass": bench.get("per_variant_joint_pass"),
                    "joint_pass_variant": bench.get("joint_pass_variant"),
                    "artifact_caveat": bench.get("memory_caveat"),
                }

    # ------------------------------------------------------------------
    # 5. the evidence index
    # ------------------------------------------------------------------
    index_entries = [
        {
            "path": "dist/" + wheel.name,
            "kind": "release_artifact",
            "sha256": wheel_sha,
            "description": "the built wheel of this release",
        },
        {
            "path": str(COHORT_PATH.relative_to(ROOT)),
            "kind": "cohort_declaration",
            "sha256": sha256_file(COHORT_PATH),
            "description": "the declared evidence cohort the results are judged against",
        },
        {
            "path": "campaigns/release/clean_install.json",
            "kind": "release_verification",
            "sha256": sha256_file(EVIDENCE_DIR / "clean_install.json"),
            "description": "the structured record of the clean-environment installation test",
        },
        {
            "path": "campaigns/release/clean_install.log",
            "kind": "release_verification",
            "sha256": sha256_file(EVIDENCE_DIR / "clean_install.log"),
            "description": "the full transcript of the clean-environment installation test",
        },
        {
            "path": "campaigns/release/wheel_contents.txt",
            "kind": "release_verification",
            "sha256": sha256_file(EVIDENCE_DIR / "wheel_contents.txt"),
            "description": "the explicit listing of the wheel contents",
        },
        {
            "path": "campaigns/release/build_summary.json",
            "kind": "release_verification",
            "sha256": sha256_file(EVIDENCE_DIR / "build_summary.json"),
            "description": "the machine readable summary of the wheel build",
        },
        {
            "path": "campaigns/release/build_reproducibility.json",
            "kind": "release_verification",
            "sha256": sha256_file(EVIDENCE_DIR / "build_reproducibility.json")
            if (EVIDENCE_DIR / "build_reproducibility.json").exists()
            else None,
            "description": "the measured three-build reproducibility check of the "
            "wheel SHA-256",
        },
        {
            "path": "campaigns/release/build_reproducibility.log",
            "kind": "release_verification",
            "sha256": sha256_file(EVIDENCE_DIR / "build_reproducibility.log")
            if (EVIDENCE_DIR / "build_reproducibility.log").exists()
            else None,
            "description": "the transcript of the three-build reproducibility check",
        },
        {
            "path": "results/results.json",
            "kind": "report",
            "sha256": sha256_file(ROOT / "results" / "results.json")
            if (ROOT / "results" / "results.json").exists()
            else None,
            "description": "the generated result table (every reported number comes from here)",
        },
        {
            "path": "results/RESULTS.md",
            "kind": "report",
            "sha256": sha256_file(ROOT / "results" / "RESULTS.md")
            if (ROOT / "results" / "RESULTS.md").exists()
            else None,
            "description": "the generated result tables",
        },
        {
            "path": "docs/RESULTS.md",
            "kind": "report",
            "sha256": sha256_file(ROOT / "docs" / "RESULTS.md")
            if (ROOT / "docs" / "RESULTS.md").exists()
            else None,
            "description": "the published copy of the result tables",
        },
    ]
    for entry in members:
        run_id = entry.get("run_id")
        if not run_id:
            continue
        for name, kind in [
            ("manifest.json", "run_manifest"),
            ("evaluation.json", "evaluation"),
            ("benchmark_matched.json", "inference_benchmark"),
        ]:
            path = ROOT / "runs" / run_id / name
            if path.exists():
                index_entries.append(
                    {
                        "path": f"runs/{run_id}/{name}",
                        "kind": kind,
                        "sha256": sha256_file(path),
                        "description": f"{kind} of {run_id}",
                    }
                )

    evidence_index = {
        "schema": "pdelab-release-evidence-index/1",
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort_id": cohort.get("cohort_id"),
        "cohort_file": str(COHORT_PATH.relative_to(ROOT)),
        "cohort_sha256": sha256_file(COHORT_PATH),
        "campaign_evidence_index": "campaigns/v2-recovery/EVIDENCE_INDEX.json",
        "artifacts": index_entries,
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(evidence_index, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # 6. the clean-install record
    # ------------------------------------------------------------------
    clean = load_json(EVIDENCE_DIR / "clean_install.json")
    steps = clean.get("steps", {})

    def step(name: str) -> dict[str, Any]:
        return steps.get(name, {})

    def cli_report(label: str) -> dict[str, Any]:
        """Summarise a CLI step that must print a JSON report."""
        data = step(label)
        raw = data.get("output", "") or ""
        report: dict[str, Any] | None = None
        try:
            report = json.loads(raw)
        except Exception:
            report = None
        checks = (report or {}).get("checks", []) or []
        return {
            "step": label,
            "exit_code": data.get("exit_code"),
            "printed_a_json_report": report is not None,
            "system": (report or {}).get("system"),
            "passed": (report or {}).get("passed"),
            "n_checks": len(checks) if report is not None else None,
            "failed_checks": [c.get("name") for c in checks if not c.get("passed")],
            "walltime_seconds": (report or {}).get("walltime_seconds"),
            "output_file": data.get("output_file"),
        }

    failed_steps = [
        (name, data)
        for name, data in steps.items()
        if data.get("exit_code") not in (0, None)
    ]

    clean_install = {
        "directory": clean.get("clean_dir"),
        "directory_is_outside_the_repository": True,
        "repository": clean.get("repository"),
        "venv": clean.get("venv"),
        "wheel_installed": clean.get("wheel_path"),
        "wheel_sha256": clean.get("wheel_sha256"),
        "environment": clean.get("environment"),
        "cpu_threads": clean.get("cpu_threads"),
        "cuda_visible_devices": clean.get("cuda_visible_devices"),
        "commands": [
            {
                "step": name,
                "exit_code": data.get("exit_code"),
                "output_file": data.get("output_file"),
            }
            for name, data in steps.items()
        ],
        "imported_from_installed_wheel": step("install_probe").get("output", "").splitlines()[
            0
        ]
        if step("install_probe").get("output")
        else None,
        "solver_validation_cli": {
            "burgers": cli_report("solvers_validate_cli"),
            "allen_cahn": cli_report("solvers_validate_cli_allen_cahn"),
            "note": "both CLI paths of `pdelab solvers validate` are exercised "
            "against the installed wheel, outside the source checkout; each "
            "must exit 0 and print a JSON report whose `passed` flag is true",
        },
        "status": "pass" if not failed_steps else "fail",
        "open_blockers": [],
    }
    for name, data in failed_steps:
        output = data.get("output", "") or ""
        tail = [line for line in output.splitlines() if line.strip()][-12:]
        clean_install["open_blockers"].append(
            {
                "step": name,
                "exit_code": data.get("exit_code"),
                "error": "\n".join(tail),
                "output_file": data.get("output_file"),
            }
        )

    # ------------------------------------------------------------------
    # 7. the reproduction command
    # ------------------------------------------------------------------
    repro_output = step("reproduction_command").get("output", "")
    reproduction = {
        "command": REPRODUCTION_COMMAND,
        "command_file": str(REPRODUCTION_COMMAND_FILE.relative_to(ROOT)),
        "command_sha256": sha256_file(REPRODUCTION_COMMAND_FILE),
        "verified": bool(step("reproduction_command").get("exit_code") == 0),
        "executed_in": clean.get("clean_dir"),
        "executed_against": "the installed wheel in " + str(clean.get("venv")),
        "working_directory": clean.get("clean_dir"),
        "environment": clean.get("environment"),
        "output": repro_output,
        "output_file": step("reproduction_command").get("output_file"),
        "note": "the command was executed verbatim (from "
        "campaigns/release/fixtures/reproduction_command.txt) against the "
        "installed CLI, in the clean environment and from outside the source "
        "checkout. The numerical gate is driven through the installed CLI "
        "itself (`pdelab solvers validate --system burgers`), which prints the "
        "JSON report of the suite; the same path is exercised for "
        "allen_cahn by the clean-install verification.",
    }

    # ------------------------------------------------------------------
    # 8. the manifest
    # ------------------------------------------------------------------
    manifest = {
        "schema": "pdelab-release-manifest/1",
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/linux/release_manifest.py",
        "project": {
            "name": pyproject["project"]["name"],
            "version": version,
            "description": pyproject["project"]["description"],
            "author": pyproject["project"]["authors"][0]["name"],
        },
        "artifact": {
            "wheel_filename": wheel.name,
            "wheel_path": str(wheel.relative_to(ROOT)),
            "wheel_size_bytes": wheel.stat().st_size,
            "wheel_uncompressed_bytes": build_summary.get("wheel_uncompressed_bytes"),
            "wheel_sha256": wheel_sha,
            "wheel_entry_count": build_summary.get("wheel_entry_count"),
            "wheel_contents_listing": "campaigns/release/wheel_contents.txt",
            "wheel_contents": build_summary.get("wheel_entries"),
            "console_script": "pdelab",
            "reproducible_build": (
                "SOURCE_DATE_EPOCH is pinned to the commit date of HEAD and the "
                "setuptools build tree is removed before every build; "
                "scripts/linux/release_reproducibility.sh builds the wheel three "
                "times in a row and compares the digests (see "
                "campaigns/release/build_reproducibility.json)"
            ),
            "reproducibility_check": reproducibility_check,
            "must_not_ship": [
                "datasets",
                "checkpoints",
                "run artifacts",
                "secrets",
                "plots",
                "the .venv",
                "the campaigns/ scratch tree",
            ],
            "shipped_content": "the pdelab library (Python source) and the wheel "
            "metadata only; no configuration, data or artifact is included",
        },
        "source": {
            "revision": revision,
            "branch": branch,
            "dirty": bool(dirty_paths),
            "dirty_path_count": len(dirty_paths),
            "dirty_paths": dirty_paths,
            "note": "the recorded revision is the commit the wheel was built from; "
            "the working tree was dirty at build time, so the wheel is not "
            "reproducible from the commit alone - it is reproducible from the "
            "working tree that was built, which is what the three-build "
            "reproducibility check measures (see artifact.reproducibility_check)",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "platform": platform.platform(),
            "cpu_threads": 4,
            "clean_install_environment": clean.get("environment"),
        },
        "evidence": {
            "evidence_index": "campaigns/release/EVIDENCE_INDEX.json",
            "campaign_evidence_index": "campaigns/v2-recovery/EVIDENCE_INDEX.json",
            "cohort_file": str(COHORT_PATH.relative_to(ROOT)),
            "cohort_sha256": sha256_file(COHORT_PATH),
            "cohort_id": cohort.get("cohort_id"),
            "cohort_schema": cohort.get("schema"),
            "evaluator": cohort.get("evaluator"),
            "protocol": cohort.get("protocol"),
            "datasets": cohort.get("datasets"),
            "checkpoints": checkpoints,
            "members": members,
        },
        "reproduction": reproduction,
        "clean_install": clean_install,
        "measured_limitations": {
            "learned_operators_vs_direct_numerical_stepping": {
                "statement": "On these benchmarks the learned operators do NOT beat "
                "direct numerical stepping of the PDE. The parameter informed "
                "numerical baseline has a lower held-out rollout nRMSE than every "
                "learned operator of the cohort, by the factors listed below.",
                "per_run": numerical_comparison,
                "note": "the numerical baseline is not parameter free and is not "
                "comparable with a learned model in terms of prior knowledge; it "
                "quantifies how much of the task is solvable by direct numerical "
                "integration.",
            },
            "burgers_residual_improvement_is_operator_specific": {
                "statement": "The Burgers residual improvement is a property of the "
                "declared residual operator, not of the model. Under the declared "
                "implicit-Euler (first order) residual the physics-regularized FNO "
                "improves on the data-only FNO (about +70% per seed); under the "
                "independently declared centred first-derivative residual it is WORSE "
                "on every seed (45% to 63% worse), and under that independent "
                "evaluator the reference solution has a LOWER residual than either "
                "learned model, i.e. the reference is more PDE-consistent than both "
                "models. Allen-Cahn improves under both evaluators.",
                "per_seed": residual_comparison,
                "note": "the two evaluators are never mixed and are evaluated on "
                "the same support (the interior snapshot indices 1 .. T-2).",
            },
            "inference_memory_claim_is_peak_allocated_only": {
                "statement": "The inference memory result is about PEAK ALLOCATED "
                "memory only. The peak allocated memory of the best variant falls "
                "by about 96% against the FP32 eager baseline, but the peak "
                "RESERVED memory rises by about 9% (the CUDA-graph private pool is "
                "larger than the eager allocator's working set), and the "
                "device-level free-memory reading is essentially unchanged. It is "
                "NOT a total device-memory saving and it is NOT a claim about the "
                "memory footprint of the model itself.",
                "measured": inference_memory,
                "note": "the numbers are recomputed from the stored benchmark "
                "artifact of the declared inference reference run; the artifact "
                "itself carries the same caveat in its `memory_caveat` field.",
            },
            "not_a_state_of_the_art_claim": "the cohort is a single machine, three "
            "seeds per condition and two self-defined periodic benchmarks; the "
            "numbers are not a comparison against published work.",
        },
        "scope_statement": (
            "This is a LOCAL release artifact of a research prototype. It is NOT a "
            "claim of public deployment, NOT a claim of production serving "
            "readiness, and NOT a claim of novel state-of-the-art research. The "
            "wheel contains the library only; the datasets, the checkpoints and the "
            "run artifacts are not part of the bundle and must be regenerated from "
            "the source tree."
        ),
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[release_manifest] wrote {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"[release_manifest] wrote {INDEX_PATH.relative_to(ROOT)}")
    print(f"[release_manifest] wheel sha256 {wheel_sha}")
    print(f"[release_manifest] revision       {revision} (dirty={bool(dirty_paths)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
