# Windows packaging

The supported deliverable is a self-contained PyInstaller **onedir** build.
`onedir` is preferred for this analysis application because it starts faster,
keeps native image libraries inspectable, and makes antivirus false positives
less likely than a single opaque archive.

## Reproducible build

On a clean Windows machine with Python 3.11 or newer:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ui,dev]"
.\scripts\build_windows.ps1
```

The script (including the default invocation, not only `-Clean`):

1. removes only the project-local disposable `build\` and `dist\` directories;
2. runs the deterministic demo/test smoke checks;
3. invokes PyInstaller with `visual_continuity_lab.spec`;
4. copies README, docs, schemas, and a demo manifest beside the executable;
5. creates `dist\VisualContinuityLab-0.1.0-win64.zip`.

The build script never touches source images or user data outside this project.

## Manual PyInstaller invocation

```powershell
python -m PyInstaller --noconfirm --clean visual_continuity_lab.spec
```

If the UI dependency is intentionally omitted, use the CLI-only target from
the script (`-CliOnly`) and verify that the resulting executable still
supports folder/video analysis and JSON export.

## First-run verification

Run these from the unpacked distribution directory:

```powershell
.\VisualContinuityLab.exe --demo --output .\demo-output
Test-Path .\demo-output\analysis.json
```

Open the generated report and check that at least one high-severity synthetic
event is present. Then import a small real image folder and verify that input
files remain unchanged.

## Common failures

- **Missing Qt DLL:** install the `[ui]` extra in the same interpreter used by
  PyInstaller, then rebuild with `--clean`.
- **OpenCV import error:** ensure `opencv-python-headless` and the active Python
  architecture (normally 64-bit) match.
- **No images discovered:** use absolute paths and check supported extensions
  (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`).
- **Video cannot open:** provide a video codec supported by the local OpenCV
  build, or extract frames externally and analyze the resulting folder.

## Release checklist

- [ ] `pytest` passes in a clean virtual environment.
- [ ] deterministic demo emits expected issue types.
- [ ] executable starts without Python installed.
- [ ] import/export schema validates with a JSON Schema validator.
- [ ] README, docs, schema, and version metadata are included in the archive.
- [ ] a real small sequence was analyzed without modifying source files.
