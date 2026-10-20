"""Compare the training histories of two runs that reported the same
best validation loss, to confirm they are genuinely independent runs."""

import json
from pathlib import Path

RUNS = Path("/home/atulp/projects/pdelab/runs")

ids = ["allen_cahn_fno2d_data_only_seed1", "allen_cahn_fno2d_data_only_seed2"]
histories = {}
for i in ids:
    m = json.loads((RUNS / i / "manifest.json").read_text())
    histories[i] = (m.get("training") or {}).get("history") or []

a, b = histories[ids[0]], histories[ids[1]]
print(f"epochs: {len(a)} vs {len(b)}")
print(f"{'epoch':>6s} {'val_total A':>16s} {'val_total B':>16s} {'val_data A':>14s} "
      f"{'val_data B':>14s} {'train_data A':>15s} {'train_data B':>15s}")
for ha, hb in list(zip(a, b))[:8] + list(zip(a, b))[-4:]:
    print(f"{ha['epoch']:>6d} {ha['val']['total']:>16.8g} {hb['val']['total']:>16.8g} "
          f"{ha['val']['data']:>14.8g} {hb['val']['data']:>14.8g} "
          f"{ha['train']['data']:>15.8g} {hb['train']['data']:>15.8g}")

identical = all(
    ha["val"]["total"] == hb["val"]["total"] for ha, hb in zip(a, b)
)
print(f"\nval total curves bitwise identical: {identical}")
best_a = min(a, key=lambda h: h["val"]["total"])
best_b = min(b, key=lambda h: h["val"]["total"])
print(f"best A: epoch {best_a['epoch']} val={best_a['val']['total']!r}")
print(f"best B: epoch {best_b['epoch']} val={best_b['val']['total']!r}")
