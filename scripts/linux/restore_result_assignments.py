"""Restore the `result =` assignments that were over-applied, keeping
only the genuinely unused one removed."""

import re
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")
p = ROOT / "tests/integration/test_training_and_resources.py"
text = p.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)

# Re-assign `result` everywhere it is used, except on the line that
# genuinely does not use it.
for i, line in enumerate(lines):
    if re.match(r"^\s+trainer\.train\(\)\s*$", line):
        indent = re.match(r"^(\s*)", line).group(1)
        # look ahead: is `result` used within the next 12 lines of this block?
        block = "".join(lines[i : i + 14])
        if re.search(r"\bresult\b", block):
            lines[i] = f"{indent}result = trainer.train()\n"

p.write_text("".join(lines), encoding="utf-8")
print("restored result assignments")
