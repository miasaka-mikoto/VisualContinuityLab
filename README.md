# Visual Continuity Lab

**AI 图像角色一致性与连续性实验室**

Visual Continuity Lab (VCL) is a local, data-driven inspection tool for image
sequences, storyboard frames, character turnarounds, and generated animation.
It measures *change between frames* and explains where continuity starts to
drift. It is a diagnostic lab, not a video editor and not a generative model.

The development/demo path is fully local: no image-generation API and no paid
model API are required. The demo creates a deterministic mock sequence with
known colour, scale, brightness, and background defects so the detector can be
verified end to end.

## What it measures

- image size, histogram, brightness, contrast, colour distribution and
  perceptual hash
- edge maps, feature matching and optical-flow-derived motion signals
- a user-locked character region (ROI), with face/hair/costume/position/scale
  continuity dimensions where the selected detector supports them
- adjacent-frame drift and master-reference drift
- brightness flicker, camera/composition changes, background drift and unusual
  character displacement
- issue records with frame range, severity, evidence and notes
- a timeline and explainable sequence-health dimensions rather than a single
  opaque score

## Quick start (source)

```powershell
py -3.11 -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ui,dev]"
python -m visual_continuity_lab --demo
```

For headless analysis/CI, `.[dev]` is sufficient; the `[ui]` extra enables
the PySide6 desktop shell (a Tk fallback remains available).

The installed console alias is equivalent:

```powershell
vclab-demo analyze --input demo/mock_sequence --output artifacts/demo
```

If your checkout uses the direct launcher, the equivalent is:

```powershell
python run_visual_continuity_lab.py --demo
```

The `--demo` run writes an analysis report and JSON export under
`artifacts/demo/` (the exact output directory can be overridden with
`--output`). For a real sequence, choose a folder or a video in the desktop UI,
lock a character ROI, select a master reference, then run **Analyze**.

## Windows executable

The supported packaging route is PyInstaller. From a Windows PowerShell prompt:

```powershell
.\scripts\build_windows.ps1
```

The script creates `dist\VisualContinuityLab\VisualContinuityLab.exe` and a
zip archive under `dist\`. See [docs/PACKAGING.md](docs/PACKAGING.md) for
clean-build, offline, and troubleshooting notes.

## Project layout

```text
visual-continuity-lab/
├─ visual_continuity_lab/       # stable public API, CLI and UI namespace
├─ vclab/                       # implementation namespace (kept for compatibility)
├─ scripts/                     # local run, demo and Windows packaging helpers
├─ schemas/                     # stable interchange contracts
├─ docs/                        # architecture, usage, QA and packaging notes
├─ tests/                       # deterministic unit/integration tests
└─ artifacts/                   # generated demo reports/screenshots (ignored)
```

The interchange contract in
[`schemas/frameforge_interchange.schema.json`](schemas/frameforge_interchange.schema.json)
is intentionally independent of FrameForge source code. VCL can import/export
shot, frame, and continuity result data through that contract.

## Design principles

1. **Evidence over mystery scores.** Every severity should be traceable to
   measurements and thresholds.
2. **Data-driven thresholds.** Analysis settings belong in JSON/YAML config,
   not in hard-coded UI callbacks.
3. **Deterministic local verification.** The mock sequence and tests must run
   without a network or model API.
4. **Non-destructive inspection.** Input images are never modified.
5. **Loose integration.** FrameForge exchange is JSON only; VCL remains a
   standalone project.

## Status

This repository is the standalone implementation and its reproducible demo.
Use the generated report as a smoke test before packaging a Windows build.
For known limitations and extension points, see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
