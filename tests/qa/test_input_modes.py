"""Input-boundary checks for stills, folders, frame sequences and video."""

from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


def _core():
    for name in ("visual_continuity_lab.core", "vclab.core"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    pytest.fail("continuity core package is unavailable")


def _frames(root: Path, count: int = 4) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for number in range(1, count + 1):
        image = Image.new("RGB", (96, 64), (20 + number * 8, 30, 45))
        # A moving, high-contrast block makes the video and optical-flow path
        # meaningful without any generated/image-model dependency.
        pixels = np.asarray(image).copy()
        x0 = 8 + number * 7
        pixels[16:48, x0 : x0 + 18] = (220, 120, 60)
        path = root / f"frame_{number:04d}.png"
        Image.fromarray(pixels).save(path)
        paths.append(path)
    return paths


def test_still_folder_and_explicit_frame_sequence_loaders(tmp_path: Path) -> None:
    core = _core()
    paths = _frames(tmp_path / "frames", 3)
    one = core.sequence_from_image(paths[0])
    assert len(one.frames) == 1
    explicit = core.sequence_from_paths([paths[2], paths[0], paths[1]])
    # The public loader defines deterministic natural ordering even when the
    # caller supplies paths in another order.
    assert [Path(frame.path).name for frame in explicit.frames] == [p.name for p in paths]
    folder = core.sequence_from_folder(tmp_path / "frames", fps=12)
    assert len(folder.frames) == 3
    assert folder.frames[1].timestamp == pytest.approx(1 / 12)
    assert folder.frames[0].frame_id


def test_video_extraction_and_source_dispatch(tmp_path: Path) -> None:
    core = _core()
    frame_root = tmp_path / "input_frames"
    paths = _frames(frame_root, 4)
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (96, 64)
    )
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV video writer unavailable in this environment")
    try:
        for path in paths:
            rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert rgb is not None
            writer.write(rgb)
    finally:
        writer.release()

    extracted = core.sequence_from_video(video_path, output_dir=tmp_path / "extracted")
    assert len(extracted.frames) == 4
    assert extracted.metadata.get("source_type") == "video"
    assert extracted.frames[0].timestamp == pytest.approx(0.0)

    dispatched = core.sequence_from_source(video_path, output_dir=tmp_path / "dispatched")
    assert len(dispatched.frames) == 4
    assert dispatched.metadata.get("source_type") == "video"


def test_manifest_json_dispatch_resolves_relative_frame_files(tmp_path: Path) -> None:
    """A Frame Sequence/shot manifest is a first-class input source."""

    from scripts.mock_sequence import generate_mock_sequence

    core = _core()
    root = tmp_path / "manifest"
    generate_mock_sequence(root, frame_count=5)
    sequence = core.sequence_from_source(root / "sequence.json")
    assert len(sequence.frames) == 5
    assert all(Path(frame.path).is_file() for frame in sequence.frames)
    assert sequence.frames[0].metadata.get("roi_boxes")


def test_shot_dictionary_loader_preserves_shot_and_frame_metadata(tmp_path: Path) -> None:
    core = _core()
    paths = _frames(tmp_path / "shot_frames", 2)
    shot = {
        "shotId": "shot-qa-01",
        "characters": [{"characterId": "character-qa"}],
        "frames": [
            {"frameId": "shot-f001", "frameNumber": 1, "path": str(paths[0]), "timestamp": 0.0},
            {"frameId": "shot-f002", "frameNumber": 2, "path": str(paths[1]), "timestamp": 0.125},
        ],
    }
    sequence = core.sequence_from_shot(shot)
    assert sequence.sequence_id == "shot-qa-01"
    assert sequence.shot == "shot-qa-01"
    assert [frame.frame_id for frame in sequence.frames] == ["shot-f001", "shot-f002"]
    assert sequence.frames[1].timestamp == pytest.approx(0.125)
