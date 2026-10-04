"""FrameForge JSON compatibility regression checks."""

from __future__ import annotations

from pathlib import Path

from scripts.mock_sequence import generate_mock_sequence
from vclab.core import (
    analyze_sequence,
    export_frameforge,
    import_frameforge,
    sequence_from_folder,
)


def test_frameforge_export_import_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    generate_mock_sequence(root)
    sequence = sequence_from_folder(root / "frames")
    report = analyze_sequence(sequence, master_index=0)

    payload = export_frameforge(sequence, report)
    assert payload["schemaVersion"] == "1.0"
    assert payload["shots"][0]["frames"]
    assert payload["shots"][0]["frames"][0]["frameId"]
    assert "continuity" in payload["shots"][0]["frames"][1]

    restored_sequence, restored_report = import_frameforge(payload)
    assert restored_sequence.sequence_id == sequence.sequence_id
    assert len(restored_sequence.frames) == len(sequence.frames)
    assert restored_report is not None
    assert len(restored_report.adjacent_results) == len(report.adjacent_results)
    # Evidence is an array in the external contract and remains available
    # after conversion to the internal mapping form.
    assert restored_report.issues
    assert "items" in restored_report.issues[0].evidence
