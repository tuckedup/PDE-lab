"""Apply the remaining lint fixes (loop-variable binding and unused
names) without shell-quoting fragility."""

import re
from pathlib import Path

ROOT = Path("/home/atulp/projects/pdelab")

FIXES = [
    ("scripts/linux/receipt_check.py",
     [(r"^for i in range\(16\):$", "for _i in range(16):")]),
    ("scripts/linux/verify_v2_data.py",
     [(r"^for system, dev_seed, test_seed in ",
       "for system, _dev_seed, _test_seed in ")]),
    ("scripts/linux/verify_audit.py",
     [(r"^(    f_plus = np\.where\(f > 0\.0, f, 0\.0\))$",
       r"\1  # noqa: F841 - kept for the record")]),
    ("tests/integration/test_training_and_resources.py",
     [(r"^(\s+)result = trainer\.train\(\)$", r"\1trainer.train()")]),
    ("campaigns/audit-verification/check_evaluation_handoff.py",
     [(r"^from pdelab\.", "from pdelab.  # noqa: E402")]),
]

for rel, subs in FIXES:
    p = ROOT / rel
    if not p.exists():
        print(f"skip (missing): {rel}")
        continue
    text = p.read_text(encoding="utf-8")
    orig = text
    for pat, rep in subs:
        text = re.sub(pat, rep, text, flags=re.M)
    if text != orig:
        p.write_text(text, encoding="utf-8")
        print("fixed", rel)
    else:
        print("unchanged", rel)
