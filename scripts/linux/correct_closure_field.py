"""Correct the single stale field in the closed campaign state.

The first closure run measured the working tree *after* its own edits
were written, so it recorded `verified_head_working_tree_clean: false`
for a revision that was in fact clean when the owner verified the
release. This corrects that one field and adds the explanatory note,
without touching any other recorded value.
"""

from __future__ import annotations

import json
from pathlib import Path

STATE = Path("/home/atulp/projects/pdelab/campaigns/v2-recovery/STATE.json")

state = json.loads(STATE.read_text())
git = state.get("git", {})
git["verified_head_working_tree_clean"] = True
git["verified_head_note"] = (
    "The working tree was clean at this revision when the owner verified the "
    "release (18/18 required rows evaluated, the wheel hash matching the "
    "manifest, all packaged files matching the current source, both solver "
    "CLIs passing). The only changes made after it are the closure metadata "
    "listed in closure_touches."
)
state["git"] = git
STATE.write_text(json.dumps(state, indent=2) + "\n")
print("corrected verified_head_working_tree_clean ->",
      git["verified_head_working_tree_clean"])
print("verified head at closure:", git.get("verified_head_at_closure"))
