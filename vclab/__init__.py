"""Visual Continuity Lab.

The public API lives in :mod:`vclab.core`.  The package intentionally keeps
the analysis engine independent from the desktop UI so it can be used from a
CLI, notebooks, or another application (for example FrameForge).
"""

from .core.analyzer import Analyzer, AnalyzerConfig, ContinuityAnalyzer
from .core.models import (
    CharacterRegion,
    ContinuityReport,
    Drift,
    DriftResult,
    Frame,
    FrameRecord,
    HealthScores,
    Issue,
    ROI,
    Sequence,
)
from .core.sequence import FrameSequence

__all__ = [
    "ContinuityAnalyzer",
    "Analyzer",
    "AnalyzerConfig",
    "ContinuityReport",
    "DriftResult",
    "Drift",
    "Frame",
    "CharacterRegion",
    "FrameSequence",
    "FrameRecord",
    "HealthScores",
    "Issue",
    "ROI",
    "Sequence",
]
