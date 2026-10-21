"""End-to-end smoke test of the dataset-facing path with the shipped
configuration (a temporary directory, the smoke profile)."""

import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

from pdelab.config import PDESystemConfig, load_yaml, project_root
from pdelab.data.generate import DatasetGenerator

root = project_root()
for name in ("burgers", "allen_cahn"):
    cfg = load_yaml(root / "configs" / "pdes" / f"{name}.yaml")
    system = PDESystemConfig.from_dict(cfg, source=str(root / "configs/pdes"))
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        gen = DatasetGenerator(
            system=system, out_dir=tmp, n_train=1, n_val=1, n_test=1,
            seed=7, profile="smoke",
        )
        manifest = gen.generate()
        elapsed = time.time() - t0
        rec = manifest["trajectories"][0]
        with h5py.File(Path(tmp) / f"{name}_train.h5", "r") as fh:
            u = fh["u"][:]
        print(f"=== {name} ===")
        print(f"  written shape            : {u.shape}")
        print(f"  requested observation times: {rec['observation_times'][:3]} ... "
              f"({len(rec['observation_times'])} entries)")
        print(f"  actual observation times  : {rec['actual_observation_times'][:3]} ... "
              f"({len(rec['actual_observation_times'])} entries)")
        print(f"  max |actual - requested|  : "
              f"{np.max(np.abs(np.array(rec['actual_observation_times']) - np.array(rec['observation_times']))):.3e}")
        print(f"  observation grid         : {system.observation_grid}")
        print(f"  reference grid           : {system.reference_grid}")
        print(f"  n_ref_steps / ref_dt     : {rec['n_ref_steps']} / {rec['ref_dt']:.4e}")
        print(f"  all finite               : {bool(np.all(np.isfinite(u)))}")
        print(f"  wall time for 3 traj     : {elapsed:.2f} s")
        print(f"  manifest scheduler       : "
              f"{manifest['observation_scheduler']['observation_scheduler']}")
        print(f"  manifest time-stepping   : "
              f"{manifest['time_stepping_policy']['internal_step_rule'][:70]}...")
