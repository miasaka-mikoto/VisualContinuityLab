"""Installed launcher for Visual Continuity Lab.

The wrapper keeps the command line independent from the desktop toolkit:
``--smoke`` is headless, ``--demo`` runs the deterministic local analysis, and
all other arguments are passed to the desktop launcher.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    # ``vclab-demo`` and source-tree users may call the demo CLI with its
    # explicit subcommands.  Keep that path independent from Qt as well.
    if raw_args and raw_args[0] in {"generate", "analyze", "run"}:
        from scripts.analyze_demo import main as demo_main

        return int(demo_main(raw_args) or 0)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--help", action="store_true")
    known, remaining = parser.parse_known_args(raw_args)
    if known.help:
        from vclab.ui.app import run

        return int(run(["--help", *remaining]) or 0)
    if known.demo:
        from scripts.analyze_demo import main as demo_main

        # With no explicit input, generate the deterministic demo first.  An
        # explicit ``--input`` keeps the flag useful for a real sequence.
        has_input = any(arg == "--input" or arg.startswith("--input=") for arg in remaining)
        if has_input:
            demo_args = ["analyze", *remaining]
        else:
            demo_args = ["run", "--demo", "demo/mock_sequence", *remaining]
        return int(demo_main(demo_args) or 0)
    from vclab.ui.app import run

    forwarded = ([] if not known.smoke else ["--smoke"]) + remaining
    return int(run(forwarded) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
