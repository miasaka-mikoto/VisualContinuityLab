# Visual Continuity Lab — Validation Report

Build: `0.1.0`  
Date: 2026-10-04

## Automated checks

- `pytest -q` — **19 passed**
- `python -m visual_continuity_lab --smoke` — passed (headless adapter)
- `python -m vclab.ui --smoke` — passed
- PySide6 offscreen UI construction and populated screenshot — passed
- Still image, folder, explicit frame sequence, MP4 extraction/source dispatch — passed
- FrameForge JSON export/import round-trip — passed
- PyInstaller onedir build and packaged `--smoke` — passed on the Linux build host

## Deterministic demo findings

The local mock sequence contains deliberate defects. The analyzer surfaced:

- costume colour drift: 7→8 and 10→11;
- character scale jump: 12→13→14;
- brightness/lighting flicker: 15→20 and 22→23;
- background drift: 15→20 and 22→23.

The report contains adjacent and master-reference comparisons, per-dimension
evidence, issue ranges, bad-frame recommendations, health dimensions, and a
drift timeline.

## Windows note

The current execution host is Linux, so the local PyInstaller smoke binary was
not presented as a Windows executable. On Windows, run
`scripts/build_windows.ps1`; it produces
`dist\\VisualContinuityLab\\VisualContinuityLab.exe` and a win64 zip archive.
