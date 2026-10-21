"""Parameter counts of the FNO2D architecture, before and after the repair.

The repaired spectral convolution keeps a separate block of complex
weights for the negative wave numbers of the first (non-halved) axis,
so the spectral parameter count grows.

Usage::

    .venv/bin/python campaigns/performance/fno2d_parameter_counts.py
"""

from __future__ import annotations

import json

from pdelab.models.base import count_parameters
from pdelab.models.fno import FNO2d

CONFIGS = [
    ("registry default (modes 8x8, width 16, 4 layers)",
     dict(modes_x=8, modes_y=8, width=16, n_layers=4)),
    ("allen_cahn experiment config (modes 8x8, width 16, 4 layers)",
     dict(modes_x=8, modes_y=8, width=16, n_layers=4)),
    ("pilot (modes 4x4, width 8, 2 layers)",
     dict(modes_x=4, modes_y=4, width=8, n_layers=2)),
]


def main() -> int:
    out = []
    for label, kwargs in CONFIGS:
        model = FNO2d(**kwargs)
        counts = count_parameters(model)
        entry = {"configuration": label, **kwargs, **counts}
        out.append(entry)
    print(json.dumps(out, indent=2))
    print()
    print(f"{'configuration':60s} {'total':>9s} {'complex':>9s} {'real':>9s}")
    print("-" * 90)
    for e in out:
        print(f"{e['configuration']:60s} {e['total']:>9d} {e['complex']:>9d} {e['real']:>9d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
