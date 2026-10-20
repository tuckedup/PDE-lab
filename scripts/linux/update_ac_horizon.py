"""Update the Allen-Cahn experiment configs to the revised horizon.

The observation time step changes from 0.05 to 0.1 to match the revised
system configuration (T = 3.0). The training budget in *steps* is
unchanged, so the number of gradient steps per epoch changes with the
number of windows; the epoch count is kept so that the optimisation
budget stays comparable to the 1D runs.
"""

import re
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab/configs/experiments")

for path in sorted(ROOT.glob("allen_cahn_*.yaml")):
    text = path.read_text(encoding="utf-8")
    new = re.sub(r"^observation_dt: .*$", "observation_dt: 0.1", text, flags=re.M)
    if new != text:
        path.write_text(new, encoding="utf-8")
        print("updated", path.name)
    else:
        print("unchanged", path.name)
