"""Record the saved session transcripts in the closed campaign state.

Adds a pointer to the durable session exports so that a restart can
find them. It does not change any recorded spend, result or verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

STATE = Path("/home/atulp/projects/pdelab/campaigns/v2-recovery/STATE.json")

state = json.loads(STATE.read_text())
state["session_records"] = {
    "opencode_session": {
        "session_id": "ses_eaa0093b7ffenKjye7spYnBw9S",
        "title": "PDELab Physics-Constrained Neural Operator Setup",
        "exported_to": "C:/Users/atulp/Documents/PDELab-sessions/"
                       "pdelab-opencode-session-ses_eaa0093b7ffenKjye7spYnBw9S.json",
        "export_bytes": 10221790,
        "export_sha256": (
            "a2352a652d7c156dcdd980df58aaee269fb11e1a5539d13b8004dcc5fbaea93d"
        ),
        "note": "The export is large (10 MB) and is kept outside the repository "
                "so the wheel and the source tree stay small.",
    },
    "audit_session_outputs": {
        "location": "C:/Users/atulp/Documents/Codex/2026-10-21/"
                    "this-is-the-full-prompt-i/outputs/",
        "files": [
            "PDELab-audit-and-recovery-plan.md",
            "PDELab-OpenCode-continuation-prompt.md",
            "PDELab-audit-evidence.json",
            "PDELab-steer-current-run.txt",
        ],
        "note": "Written by the owner's read-only audit session; not modified "
                "by the implementation campaign.",
    },
    "repository_state": {
        "branch": "master",
        "closure_commit": "1e8120c (administrative closure; metadata only)",
        "remote": "none - nothing was pushed",
    },
}
STATE.write_text(json.dumps(state, indent=2) + "\n")
print("session records written")
