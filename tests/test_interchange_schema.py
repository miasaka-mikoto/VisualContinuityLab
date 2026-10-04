"""Smoke checks for the standalone FrameForge interchange contract."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_schema_and_example_are_valid_json() -> None:
    schema = json.loads((ROOT / "schemas/frameforge_interchange.schema.json").read_text(encoding="utf-8"))
    example = json.loads((ROOT / "schemas/frameforge_interchange.example.json").read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("draft/2020-12/schema")
    assert schema["properties"]["schemaVersion"]["const"] == "1.0"
    assert example["schemaVersion"] == "1.0"
    assert example["shots"] and example["shots"][0]["frames"]


def test_example_has_stable_frameforge_keys() -> None:
    example = json.loads((ROOT / "schemas/frameforge_interchange.example.json").read_text(encoding="utf-8"))
    shot = example["shots"][0]
    assert shot["masterReferenceFrameId"] == "frame-001"
    frame = shot["frames"][0]
    assert {"frameId", "frameNumber", "status"}.issubset(frame)
    assert frame["continuity"]["master"]["severity"] in {"none", "low", "medium", "high", "unknown"}

