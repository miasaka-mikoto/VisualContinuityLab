"""Report/interchange compatibility module.

The engine's canonical report model is in :mod:`vclab.core.models`; this thin
module keeps the intuitive ``vclab.core.report`` import path available to
scripts and plugins.
"""

from .io import (
    export_frameforge,
    export_frameforge_json,
    export_json,
    import_frameforge,
    import_json,
    report_from_frameforge,
    sequence_from_frameforge,
)
from .models import ContinuityReport, HealthScores, Issue

__all__ = [
    "ContinuityReport", "HealthScores", "Issue", "export_frameforge",
    "export_frameforge_json", "export_json", "import_frameforge", "import_json",
    "report_from_frameforge", "sequence_from_frameforge",
]
