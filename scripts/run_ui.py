"""Launch the Visual Continuity Lab desktop UI from a source checkout.

Examples::

    python scripts/run_ui.py
    python scripts/run_ui.py --smoke
    python scripts/run_ui.py --tk
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from vclab.ui.app import run


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
