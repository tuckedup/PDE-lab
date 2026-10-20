"""Check that the training seed actually reaches the model.

Two things are verified:
  1. The model *initialisation* differs between different seeds. The
     seed is currently set inside Trainer.train(), which runs AFTER the
     model is constructed, so the initialisation is currently unseeded.
  2. The recorded training history differs between seeds.
"""

import json
from pathlib import Path

import torch

RUNS = Path("/home/atulp/projects/pdelab/runs")


def first_param_tensor(run_id: str) -> torch.Tensor:
    ck = torch.load(RUNS / run_id / "best.pt", map_location="cpu", weights_only=False)
    return next(iter(ck["model"].values())).flatten()[:8].clone()


def manifest(run_id: str) -> dict:
    return json.loads((RUNS / run_id / "manifest.json").read_text())


pairs = [
    "burgers_fno1d_data_only",
    "burgers_fno1d_physics",
    "burgers_unet1d_data_only",
    "allen_cahn_fno2d_data_only",
    "allen_cahn_fno2d_physics",
    "allen_cahn_unet2d_data_only",
]

print(f"{'condition':36s} {'seed0==seed1 init':>20s} {'seed1==seed2 init':>20s} "
      f"{'val loss s0/s1/s2':>34s}")
print("-" * 112)
for cond in pairs:
    ids = [f"{cond}_seed{s}" for s in (0, 1, 2)]
    if not all((RUNS / i / "best.pt").exists() for i in ids):
        print(f"{cond:36s}  (missing runs)")
        continue
    p0, p1, p2 = (first_param_tensor(i) for i in ids)
    same01 = bool(torch.equal(p0, p1))
    same12 = bool(torch.equal(p1, p2))
    vals = []
    for i in ids:
        m = manifest(i)
        hist = (m.get("training") or {}).get("history") or []
        vals.append(min((h["val"].get("total", float("inf")) for h in hist), default=float("nan")))
    print(f"{cond:36s} {str(same01):>20s} {str(same12):>20s} "
          f"{vals[0]:.6g}/{vals[1]:.6g}/{vals[2]:.6g}")
