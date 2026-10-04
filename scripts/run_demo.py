"""Small compatibility launcher for the deterministic local demo."""
from __future__ import annotations

import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path

# When invoked as ``python scripts/run_demo.py`` Python puts only the scripts
# directory on sys.path; add the checkout root so the installed-launcher shim
# can be imported without first running ``pip install -e .``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from visual_continuity_lab.__main__ import main as app_main

    # Keep the wrapper useful both as `python scripts/run_demo.py` and when a
    # caller supplies additional CLI flags.
    if "--demo" not in sys.argv[1:]:
        sys.argv.insert(1, "--demo")
    return int(app_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
