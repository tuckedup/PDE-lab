"""Rewrite the operator test module in the before-tree so that it can be
collected against the pre-repair source (the new operators do not exist
there).  The tests that need them then fail, which is the point of the
before/after comparison.
"""

import re
from pathlib import Path

DST = Path("/home/atulp/pdelab_old_head/tests/unit/test_operators.py")
SRC = Path("/home/atulp/projects/pdelab/tests/unit/test_operators.py")

NEW_NAMES = [
    "allen_cahn_residual_centered_first_derivative",
    "allen_cahn_residual_implicit_euler",
    "burgers_residual_centered_first_derivative",
    "burgers_residual_implicit_euler",
    "dealias_1_2",
    "time_derivative_centered_first_order",
    "time_derivative_forward_first_order",
]

text = SRC.read_text()
m = re.search(r"^from pdelab\.physics\.operators import \([^)]*\)", text, re.MULTILINE)
assert m is not None
block = m.group(0)
inner = block[block.index("(") + 1: block.rindex(")")].strip()
names = [line.strip().rstrip(",") for line in inner.splitlines() if line.strip()]
old_names = [n for n in names if n not in NEW_NAMES]
new_names = [n for n in names if n in NEW_NAMES]

guarded = (
    "try:\n"
    + "\n".join("    " + line for line in block.splitlines())
    + "\nexcept ImportError as _exc:  # before-variant only\n"
    + "    _MISSING = _exc\n"
    + "    from pdelab.physics.operators import (\n"
    + "".join(f"        {n},\n" for n in old_names)
    + "    )\n"
    + "".join(f"    {n} = None\n" for n in new_names)
)
text = text[: m.start()] + guarded + text[m.end():]
DST.write_text(text)
print(f"wrote {DST}")
