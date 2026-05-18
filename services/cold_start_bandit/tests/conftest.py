"""Add repo root to sys.path so ``from shared...`` and ``from app...`` resolve
when pytest is run from inside the service directory.
"""

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVICE_ROOT.parent.parent

for p in (str(REPO_ROOT), str(SERVICE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
