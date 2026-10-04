"""Application launcher and headless smoke entrypoint."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from .adapter import AnalysisAdapter


def smoke_test() -> dict:
    """Run a deterministic no-display test of loading and analysis.

    The smoke path is used by CI and by packagers to verify that importing the
    application does not require an X/Wayland display.
    """
    with tempfile.TemporaryDirectory(prefix="vclab-smoke-") as temp:
        root = Path(temp)
        for index in range(1, 5):
            image = Image.new("RGB", (256, 192), (24 + index * 12, 38, 52))
            draw = ImageDraw.Draw(image)
            size = 48 + index * 4
            draw.rectangle((112 - size // 2, 76 - size // 2, 112 + size // 2, 76 + size // 2), fill=(160, 190 - index * 12, 210))
            if index == 3:  # deliberate colour/scale perturbation
                draw.rectangle((95, 45, 160, 110), fill=(245, 80, 60))
            image.save(root / f"frame_{index:03d}.png")
        adapter = AnalysisAdapter()
        records = adapter.load_paths([str(root)])
        result = adapter.analyze(records)
        assert len(records) == 4, f"expected 4 frames, got {len(records)}"
        assert "Overall" in result.get("health", {}), "missing health score"
        assert len(result.get("drift", [])) == 3, "missing adjacent drift rows"
        return {"frames": len(records), "issues": len(result.get("issues", [])), "overall": result["health"]["Overall"]}


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="visual-continuity-lab", description="AI 图像角色一致性与连续性实验室")
    parser.add_argument("--smoke", action="store_true", help="run a headless adapter smoke test")
    parser.add_argument("--tk", action="store_true", help="force the Tkinter fallback UI")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.smoke:
        print(smoke_test())
        return 0

    if not args.tk:
        try:
            from PySide6.QtWidgets import QApplication
            from .main_window import MainWindow

            app = QApplication.instance() or QApplication([])
            window = MainWindow()
            window.show()
            return app.exec()
        except ImportError:
            pass
    from .tk_fallback import run_tk

    run_tk()
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

