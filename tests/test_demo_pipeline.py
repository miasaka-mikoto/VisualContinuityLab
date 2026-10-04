"""Regression tests for the deterministic Visual Continuity Lab demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Some pytest launchers do not put the repository root on ``sys.path`` when
# invoked via an installed entry point.  Keep this source-tree test runnable
# without requiring an editable package install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_demo import analyze_sequence, write_report
from scripts.mock_sequence import frame_spec, generate_mock_sequence


def _transitions(report: dict, metric: str) -> set[tuple[int, int]]:
    return {
        (entry["from_frame"], entry["to_frame"])
        for entry in report["adjacent_consistency"]
        if entry["metrics"][metric]["severity"] != "Low"
    }


def test_frame_specs_expose_all_demo_markers() -> None:
    assert frame_spec(8)["color_drift"] is True
    assert frame_spec(13)["scale_jump"] is True
    assert frame_spec(16)["brightness_flicker"] is True
    assert frame_spec(20)["background_change"] is True
    assert not any(frame_spec(number)["scale_jump"] for number in (12, 14))


def test_generate_mock_sequence_is_reproducible_and_manifested(tmp_path: Path) -> None:
    manifest = generate_mock_sequence(tmp_path / "sequence", frame_count=24)
    root = tmp_path / "sequence"
    assert manifest["sequence_id"] == "mock-continuity-sequence-001"
    assert manifest["frame_count"] == 24
    assert len(list((root / "frames").glob("frame_*.png"))) == 24
    assert (root / "preview_montage.png").is_file()
    assert manifest["demo_markers"] == {
        "color_drift": [8, 9, 10],
        "scale_jump": [13],
        "brightness_flicker": [16, 18],
        "background_change": [20, 21, 22],
    }
    loaded = json.loads((root / "sequence.json").read_text(encoding="utf-8"))
    assert loaded["frames"][12]["markers"]["scale_jump"] is True
    assert loaded["frames"][19]["markers"]["background_change"] is True


def test_analyzer_finds_intentional_events_and_writes_artifacts(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequence"
    report_dir = tmp_path / "report"
    generate_mock_sequence(sequence_dir)
    report = analyze_sequence(sequence_dir)

    assert report["report_type"] == "Shot Continuity Report"
    assert report["sequence"]["frame_count"] == 24
    assert report["demo_markers"]["color_drift"] == [8, 9, 10]
    # Transition boundaries are the useful evidence: the colour event begins
    # at 7→8, the scale event at 12→13, etc.
    assert (7, 8) in _transitions(report, "color_drift")
    assert (12, 13) in _transitions(report, "character_scale_jump")
    assert (15, 16) in _transitions(report, "brightness_flicker")
    assert (19, 20) in _transitions(report, "background_drift")
    assert report["sequence_health"]["overall"] <= 100
    assert report["issues"]

    paths = write_report(report, report_dir)
    for path in paths.values():
        artifact = Path(path)
        assert artifact.exists() and artifact.stat().st_size > 100
    assert "Shot Continuity Report" in (report_dir / "shot_continuity_report.html").read_text(encoding="utf-8")
