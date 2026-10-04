from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vclab.core import (
    ContinuityAnalyzer,
    ROI,
    Sequence,
    export_frameforge,
    histogram,
    optical_flow,
    sequence_from_folder,
)


def _write_frames(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for number, color in enumerate(((40, 130, 200), (220, 60, 90), (40, 130, 200)), 1):
        image = Image.new("RGB", (96, 64), (15, 20, 30))
        draw = ImageDraw.Draw(image)
        x = 25 if number != 3 else 45
        draw.rectangle((x, 12, x + 24, 54), fill=color)
        image.save(root / f"frame_{number:04d}.png")


def test_roi_pixel_conversion_and_sequence_json_roundtrip(tmp_path: Path):
    roi = ROI(0.25, 0.125, 0.5, 0.75, normalized=True, locked=True)
    assert roi.as_pixels((200, 100)) == (50, 12, 150, 88)
    sequence = Sequence.from_dict({
        "sequence_id": "s1",
        "frames": [{"frame_id": "f1", "path": "x.png", "frame_number": 1, "roi": roi.to_dict()}],
    })
    restored = Sequence.from_json(sequence.to_json())
    assert restored.sequence_id == "s1"
    assert restored.frames[0].roi.locked is True


def test_loader_is_natural_sorted_and_metrics_are_json_safe(tmp_path: Path):
    _write_frames(tmp_path)
    sequence = sequence_from_folder(tmp_path)
    assert [frame.frame_number for frame in sequence.frames] == [1, 2, 3]
    assert [Path(frame.path).name for frame in sequence.frames] == ["frame_0001.png", "frame_0002.png", "frame_0003.png"]
    image = np.asarray(Image.open(sequence.frames[0].path))
    assert len(histogram(image)) == 96
    flow = optical_flow(image, image)
    assert flow["mean_magnitude"] < 1e-4


def test_analyzer_emits_adjacent_master_timeline_issues_and_health(tmp_path: Path):
    _write_frames(tmp_path)
    sequence = sequence_from_folder(tmp_path)
    # Explicit ROI prevents the fallback detector from including the dark stage.
    sequence.set_roi(ROI(0.15, 0.1, 0.7, 0.85, locked=True), lock=True)
    report = ContinuityAnalyzer().analyze(sequence, master_index=0)
    assert len(report.adjacent_results) == 2
    assert len(report.master_results) == 2
    assert "lighting" in report.timeline
    assert report.health.sample_count == 2
    assert report.issues
    payload = report.to_dict()
    assert json.loads(json.dumps(payload))["sequence_id"] == sequence.sequence_id


def test_frameforge_export_contains_versioned_shot_and_health(tmp_path: Path):
    _write_frames(tmp_path)
    sequence = sequence_from_folder(tmp_path)
    report = ContinuityAnalyzer().analyze(sequence)
    payload = export_frameforge(sequence, report)
    assert payload["schemaVersion"] == "1.0"
    assert payload["shots"][0]["frames"]
    assert set((payload["health"] or {})) >= {"identityStability", "overall"}
