"""Verify that distinct configurations map to distinct run directories."""

from pathlib import Path

from pdelab.training.experiment import load_experiment_config, run_id_for

ROOT = Path("/home/atulp/projects/pdelab/configs/experiments")

for path in sorted(ROOT.glob("*.yaml")):
    if path.name == "main_matrix.yaml":
        continue
    cfg, _ = load_experiment_config(path)
    print(f"{path.stem:46s} -> run_id {run_id_for(cfg)}")
