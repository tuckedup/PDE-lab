"""Two consequences of the FNO2D repair, measured on the real artifacts.

1. The old FNO2D checkpoints are a **different architecture**: the
   repaired spectral convolution has two weight blocks, so the old
   state dictionaries do not load into it.
2. The benchmark artifact that was produced before the identity-bound
   protocol carries no content identity, so the report refuses it.

Usage::

    .venv/bin/python campaigns/performance/check_architecture_change.py
"""

from __future__ import annotations

import json
import sys

import torch

from pdelab.config import load_yaml, project_root
from pdelab.evaluation.metrics import evaluator_identity
from pdelab.models.registry import build_model_from_config
from pdelab.reporting.report import (
    DEFAULT_COHORT_PATH,
    assess_run_evidence,
    collect_runs,
)


def _load_old_checkpoint(run_id: str) -> dict:
    run_dir = project_root() / "runs" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["experiment"]
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model_from_config(
        {
            "architecture": cfg["architecture"],
            "history": cfg.get("history", 1),
            **{k: v for k, v in (cfg.get("model_kwargs") or {}).items()},
        },
        dimension=2 if cfg["system"] == "allen_cahn" else 1,
    )
    return {"model": model, "state": ckpt["model"], "cfg": cfg}


def main() -> int:
    print("=== 1. the old FNO2D checkpoints are a different architecture ===")
    run_id = "allen_cahn_fno2d_data_only_seed0"
    try:
        loaded = _load_old_checkpoint(run_id)
    except FileNotFoundError:
        print(f"no run {run_id} on disk; skipped")
        loaded = None
    if loaded is not None:
        keys = sorted(loaded["state"].keys())
        print("checkpoint spectral keys (old):",
              [k for k in keys if "spectral" in k][:4])
        print("model spectral keys (repaired):",
              sorted(k for k in loaded["model"].state_dict() if "spectral" in k)[:4])
        try:
            loaded["model"].load_state_dict(loaded["state"])
            print("UNEXPECTED: the old checkpoint loaded into the repaired model")
            rc = 1
        except RuntimeError as exc:
            print("the old checkpoint does NOT load into the repaired model:")
            for line in str(exc).splitlines()[:6]:
                print("   ", line)
            rc = 0

    print()
    print("=== 2. the old benchmark artifact carries no content identity ===")
    cohort = load_yaml(project_root() / DEFAULT_COHORT_PATH)
    evaluator_now = evaluator_identity()
    runs = collect_runs(project_root() / "runs")
    for run in runs:
        if run["run_id"] != "burgers_fno1d_data_only_seed0":
            continue
        for artifact in ("evaluation", "benchmark"):
            evidence = assess_run_evidence(run, cohort, evaluator_now, artifact=artifact)
            print(f"{artifact:11s} state={evidence['state']} reason={evidence['reason']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
