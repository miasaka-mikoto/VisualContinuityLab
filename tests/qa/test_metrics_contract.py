"""Contract checks for the local, model-free visual metric primitives."""

from __future__ import annotations

import importlib

import numpy as np


def _metrics():
    for name in ("visual_continuity_lab.core.image_metrics", "vclab.core.image_metrics"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    raise AssertionError("image metrics module is unavailable")


def test_basic_visual_metrics_are_json_safe_and_deterministic() -> None:
    metrics = _metrics()
    first = np.zeros((48, 64, 3), dtype=np.uint8)
    second = first.copy()
    first[10:30, 20:38] = (220, 80, 45)
    second[12:32, 23:41] = (80, 160, 220)

    assert metrics.image_size(first) == (64, 48)
    hist = metrics.histogram(first)
    assert len(hist) == 96 and all(float(value) >= 0 for value in hist)
    assert 0 <= metrics.brightness(first) <= 1
    assert 0 <= metrics.contrast(first) <= 1
    distribution = metrics.color_distribution(first)
    assert distribution and all(float(value) >= 0 for value in distribution)
    digest = metrics.perceptual_hash(first)
    assert digest == metrics.perceptual_hash(first) and digest
    edges = metrics.edge_map(first)
    assert getattr(edges, "shape", None) == (48, 64)
    assert 0 <= metrics.edge_density(first) <= 1
    assert 0 <= metrics.sharpness(first) <= 1

    matches = metrics.compare_feature_matches(first, second)
    assert 0 <= float(matches.get("score", 0.0)) <= 1
    flow = metrics.optical_flow(first, second)
    assert float(flow.get("mean_magnitude", 0.0)) >= 0

