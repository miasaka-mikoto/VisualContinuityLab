# PyInstaller onedir specification for Visual Continuity Lab.
# Build with: python -m PyInstaller --noconfirm --clean visual_continuity_lab.spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path(SPEC).parent

hiddenimports = []
for package in ("visual_continuity_lab", "vclab", "scripts"):
    try:
        hiddenimports.extend(collect_submodules(package))
    except Exception:
        # One compatibility package may be absent in a minimal CLI checkout.
        pass

datas = [
    (str(ROOT / "schemas"), "schemas"),
    (str(ROOT / "docs"), "docs"),
]
for package in ("visual_continuity_lab", "vclab"):
    try:
        datas.extend(collect_data_files(package))
    except Exception:
        pass

a_entry = ROOT / "visual_continuity_lab" / "__main__.py"
if not a_entry.exists():
    a_entry = ROOT / "vclab" / "ui" / "app.py"

a = Analysis(
    [str(a_entry)],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The demo report has a Pillow-only plotting fallback; keeping matplotlib
    # out of the Windows bundle makes builds reproducible and much smaller.
    excludes=["matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    name="VisualContinuityLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    exclude_binaries=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    # PySide6's Linux hook can contribute compatibility symlinks whose flat
    # destination clashes with a similarly named Qt binary in a one-folder
    # build.  The real library is already included via ``a.binaries``; omit
    # only these aliases.  Windows builds do not emit them, and the filtered
    # form keeps the spec reproducible on both platforms.
    [entry for entry in a.datas if entry[2] != "SYMLINK"],
    strip=False,
    upx=False,
    name="VisualContinuityLab",
)
