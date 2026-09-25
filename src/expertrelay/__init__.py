"""ExpertRelay: virtual memory for AI.

A Manager that understands every device in a cluster, keeps hot experts in
RAM and cold experts on SSD or a networked peer, predicts which expert a
model will need next, and starts moving it before the router asks for it.

See docs/limitations.md for exactly which pieces below are implemented vs.
placeholders, and docs/setup-notes.md for the reduced-model validation work
this package grew out of.
"""

from pathlib import Path

__version__ = "0.1.0"

# src/expertrelay/__init__.py -> expertrelay/ -> src/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
