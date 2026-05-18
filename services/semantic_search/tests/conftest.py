"""Add repo root + service root to sys.path so imports resolve."""

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVICE_ROOT.parent.parent

for p in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
