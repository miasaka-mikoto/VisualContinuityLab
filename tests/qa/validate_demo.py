#!/usr/bin/env python3
"""Small CI-friendly QA runner for a generated demo sequence.

Unlike pytest this emits a concise JSON summary suitable for a packaging
smoke-test log.  It intentionally uses the same public APIs as the desktop
application and exits non-zero when a planted anomaly is not observed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mock_sequence import generate_mock_sequence


def run(output: str | Path | None = None) -> dict[str, object]:
    root = Path(output) if output else Path(tempfile.mkdtemp(prefix="vcl-qa-"))
    manifest = generate_mock_sequence(root)
    try:
        from visual_continuity_lab.core import ContinuityAnalyzer, sequence_from_folder
    except ModuleNotFoundError:
        from vclab.core import ContinuityAnalyzer, sequence_from_folder

    image_root = root / "frames" if (root / "frames").is_dir() else root
    sequence = sequence_from_folder(image_root)
    report = ContinuityAnalyzer().analyze(sequence, master_index=0)
    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    issues = data.get("issues", []) or []
    health = data.get("health", {}) or {}
    timeline = data.get("timeline", {}) or {}
    if len(issues) == 0:
        raise AssertionError("no issues were emitted for the planted mock defects")
    if not timeline:
        raise AssertionError("drift timeline is empty")
    if not health:
        raise AssertionError("health dimensions are empty")
    return {
        "sequence": str(root),
        "frames": manifest["frame_count"],
        "issues": len(issues),
        "health_keys": sorted(str(k) for k in health),
        "timeline_keys": sorted(str(k) for k in timeline),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
