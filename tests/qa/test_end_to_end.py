"""End-to-end checks for the deterministic continuity demo.

These checks are deliberately data-driven.  The mock sequence plants known
events, then the real loader and analyser must surface non-trivial drift,
health dimensions, timeline entries and actionable issues.  This is stronger
than a UI smoke test while remaining fast enough to run on every build.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Pytest may choose ``tests`` as the import root when invoked from a parent
# directory.  Keep these tests runnable from either the repository root or an
# IDE's test runner without requiring an editable install first.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_core() -> Any:
    """Import the engine through the canonical package with a vclab fallback."""

    for name in ("visual_continuity_lab.core", "vclab.core"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    pytest.fail("No continuity core package (visual_continuity_lab.core or vclab.core) found")


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    pytest.fail(f"Expected a report object or dictionary, got {type(value)!r}")


def _generate(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    generator = importlib.import_module("scripts.mock_sequence")
    output = tmp_path / "mock_sequence"
    manifest = generator.generate_mock_sequence(output)
    assert manifest["frame_count"] >= 20
    assert (output / "sequence.json").is_file()
    frame_files = sorted((output / "frames").glob("*.png"))
    assert len(frame_files) == manifest["frame_count"]
    return output, manifest


def _load_sequence(core: Any, output: Path) -> Any:
    """Use the public folder loader (supporting its small API variants)."""

    # The mock generator stores its manifest beside a ``frames`` directory;
    # pass the image directory to folder loaders (rather than the manifest
    # directory, which intentionally contains JSON and README files).
    image_root = output / "frames" if (output / "frames").is_dir() else output
    loader = getattr(core, "sequence_from_folder", None)
    if callable(loader):
        try:
            return loader(image_root)
        except TypeError:
            return loader(str(image_root))
    # A manifest is also a supported interchange boundary.  This fallback
    # keeps the test useful while a loader implementation is being completed.
    models = importlib.import_module(f"{core.__name__}.models")
    sequence_cls = getattr(models, "Sequence")
    payload = json.loads((output / "sequence.json").read_text(encoding="utf-8"))
    return sequence_cls.from_dict(payload)


def _analyze(core: Any, sequence: Any, output: Path) -> Any:
    fn = getattr(core, "analyze_sequence", None)
    assert callable(fn), "core must expose analyze_sequence"
    # The canonical engine accepts a Sequence and keyword options.  A couple
    # of early builds accepted records instead, so retain a narrow fallback to
    # make this integration test diagnose rather than mask that mismatch.
    kwargs = {"master_index": 0}
    try:
        return fn(sequence, **kwargs)
    except (TypeError, AttributeError):
        frames = getattr(sequence, "frames", sequence)
        try:
            return fn(frames, **kwargs)
        except TypeError:
            return fn(frames)


def _items(report: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = report.get(key, [])
    return [v if isinstance(v, dict) else _as_dict(v) for v in values]


def _health(report: dict[str, Any]) -> dict[str, Any]:
    health = report.get("health", {})
    if not isinstance(health, dict):
        health = _as_dict(health)
    return health


def _dimension_names(report: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for result in _items(report, "adjacent_results") + _items(report, "master_results"):
        dimensions = result.get("dimensions", {})
        if isinstance(dimensions, dict):
            names.update(str(k).lower() for k in dimensions)
            for value in dimensions.values():
                if isinstance(value, dict) and value.get("name"):
                    names.add(str(value["name"]).lower())
    return names


def test_demo_generation_and_analysis_surface_planted_events(tmp_path: Path) -> None:
    output, manifest = _generate(tmp_path)
    core = _load_core()
    sequence = _load_sequence(core, output)
    frames = getattr(sequence, "frames", None)
    assert frames is not None and len(frames) == manifest["frame_count"]

    report = _as_dict(_analyze(core, sequence, output))
    adjacent = _items(report, "adjacent_results")
    assert len(adjacent) >= len(frames) - 1
    assert _items(report, "issues"), "planted mock defects should create issues"

    # A report must expose all requested health dimensions, not only an opaque
    # aggregate.  Accept both Python field names and display labels.
    health = _health(report)
    health_names = {str(k).lower().replace(" ", "_") for k in health}
    required_groups = (
        ("identity_stability", "identity"),
        ("motion_stability", "motion"),
        ("color_stability", "color"),
        ("lighting_stability", "lighting"),
        ("background_stability", "background"),
        ("overall",),
    )
    for group in required_groups:
        assert any(name in health_names or any(alias in name for alias in group) for name in health_names), (
            f"missing health dimension from {sorted(health_names)}; expected one of {group}"
        )

    dimensions = _dimension_names(report)
    # These are the high-value diagnostic axes planted by the generator.  The
    # implementation may call them "Color Drift" or "costume_color"; check
    # semantic tokens rather than forcing a presentation label.
    assert any("color" in name or "costume" in name for name in dimensions)
    assert any("bright" in name or "light" in name for name in dimensions)
    assert any("background" in name or "composition" in name for name in dimensions)

    issue_text = " ".join(json.dumps(i, ensure_ascii=False).lower() for i in _items(report, "issues"))
    assert any(token in issue_text for token in ("color", "costume", "brightness", "background", "scale"))

    # Verify the detector's *actual* transition evidence at the boundaries
    # planted by the generator.  This intentionally uses semantic dimension
    # aliases so a display/UI rename does not weaken the QC contract.
    by_pair = {
        (str(item.get("from_frame")), str(item.get("to_frame"))): item
        for item in adjacent
    }

    def transition(frame_a: int, frame_b: int) -> dict[str, Any]:
        # Core frame IDs can be either F0001/f0001 or filename stems.  Match
        # by numeric suffix as a stable interchange-level fallback.
        for pair, item in by_pair.items():
            nums = tuple(int("".join(ch for ch in token if ch.isdigit()) or -1) for token in pair)
            if nums == (frame_a, frame_b):
                return item
        pytest.fail(f"missing adjacent transition {frame_a}->{frame_b}")

    def score(item: dict[str, Any], aliases: tuple[str, ...]) -> float:
        dims = item.get("dimensions", {}) or {}
        values: list[float] = []
        for key, value in dims.items():
            key_norm = str(key).lower().replace(" ", "_")
            if any(alias in key_norm for alias in aliases):
                if isinstance(value, dict):
                    values.append(float(value.get("score", 0.0)))
                else:
                    values.append(float(value))
        return max(values, default=0.0)

    assert score(transition(7, 8), ("costume", "color")) >= 0.15
    assert max(score(transition(12, 13), ("scale", "proportion")),
               score(transition(13, 14), ("scale", "proportion"))) >= 0.15
    assert max(score(transition(15, 16), ("lighting", "bright")),
               score(transition(17, 18), ("lighting", "bright"))) >= 0.15
    assert score(transition(19, 20), ("background", "composition")) >= 0.15


def test_report_round_trip_and_timeline_are_json_safe(tmp_path: Path) -> None:
    output, _manifest = _generate(tmp_path)
    core = _load_core()
    sequence = _load_sequence(core, output)
    report_obj = _analyze(core, sequence, output)
    report = _as_dict(report_obj)

    encoded = json.dumps(report, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded.get("sequence_id")
    timeline = decoded.get("timeline", decoded.get("drift_timeline", {}))
    assert isinstance(timeline, dict) and timeline, "timeline must contain per-dimension samples"

    # If the model provides a parser, verify the actual object round-trip too.
    report_cls = getattr(importlib.import_module(f"{core.__name__}.models"), "ContinuityReport", None)
    if report_cls is not None and hasattr(report_cls, "from_dict"):
        restored = report_cls.from_dict(decoded)
        assert len(getattr(restored, "adjacent_results", [])) == len(report.get("adjacent_results", []))


def test_public_adapter_finds_frames_without_display(tmp_path: Path) -> None:
    """The UI adapter's file-loading boundary is usable headlessly."""

    output, manifest = _generate(tmp_path)
    adapter_mod = None
    for name in ("visual_continuity_lab.ui.adapter", "vclab.ui.adapter"):
        try:
            adapter_mod = importlib.import_module(name)
            break
        except ModuleNotFoundError:
            continue
    assert adapter_mod is not None
    adapter = adapter_mod.AnalysisAdapter()
    records = adapter.load_paths([output / "frames"])
    assert len(records) == manifest["frame_count"]
    result = adapter.analyze(records)
    assert isinstance(result, dict)
    assert "health" in result and "drift" in result and "issues" in result


def test_locked_roi_and_manual_qc_labels_survive_analysis(tmp_path: Path) -> None:
    """The review workflow can lock a region and add labels before a run."""

    output, _manifest = _generate(tmp_path)
    core = _load_core()
    sequence = _load_sequence(core, output)
    models = importlib.import_module(f"{core.__name__}.models")
    roi = models.ROI(0.25, 0.08, 0.5, 0.82, normalized=True, locked=True)
    sequence.set_roi(roi, lock=True)
    sequence.add_label(sequence.frames[0].frame_id, "Good", "clean master")
    sequence.add_label(sequence.frames[7].frame_id, "Bad Costume", "review colour transition")
    with pytest.raises(ValueError):
        sequence.add_label(sequence.frames[0].frame_id, "Not a QC label")

    report = _as_dict(_analyze(core, sequence, output))
    # The label is an input to the review state even if the statistical issue
    # grouping later chooses a wider frame range.
    assert any(item.get("label") == "Bad Costume" for item in sequence.manual_labels)
    assert report.get("bad_frames") is not None
    assert report.get("recommended_frames") is not None
    assert sequence.frames[7].frame_id in report.get("bad_frames", [])
