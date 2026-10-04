"""Desktop user interface for Visual Continuity Lab."""

from .app import run

try:  # Keep package importable on systems without the optional Qt dependency.
    from .main_window import MainWindow
except Exception:  # pragma: no cover
    MainWindow = None  # type: ignore[assignment]

__all__ = ["run", "MainWindow"]
