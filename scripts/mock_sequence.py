#!/usr/bin/env python3
"""Generate a deterministic synthetic sequence for Visual Continuity Lab.

The generator intentionally introduces a handful of continuity failures so the
demo can exercise the analysis pipeline without an image-generation service:

* a costume colour drift (frames 8--10),
* a character scale jump (frame 13),
* a brightness flicker (frames 16 and 18), and
* a background change (frames 20--22).

The output is a folder of ordinary PNG files plus ``sequence.json``.  The
manifest is deliberately small and stable, making it useful as a fixture in
tests and as an example import/export payload.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

try:  # Numpy is present in the desktop build; the fallback keeps this script portable.
    import numpy as np
except Exception:  # pragma: no cover - only used on a minimal Python install
    np = None


DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
DEFAULT_FRAMES = 24
DEFAULT_FPS = 12.0
SCHEMA_VERSION = "1.0"


def _clamp(value: float, low: int = 0, high: int = 255) -> int:
    return int(max(low, min(high, round(value))))


def _lerp(a: int, b: int, t: float) -> int:
    return _clamp(a + (b - a) * t)


def _gradient_background(width: int, height: int, *, changed: bool = False) -> Image.Image:
    """Create a smooth, low-noise background with a deterministic variant."""

    top = (42, 57, 94) if not changed else (80, 39, 72)
    bottom = (12, 20, 39) if not changed else (24, 56, 65)
    if np is not None:
        # Vectorized construction keeps the fixture quick enough to regenerate
        # in a test suite while retaining deterministic gradients and waves.
        ys = np.linspace(0.0, 1.0, max(1, height), dtype=np.float32)[:, None]
        xs = np.arange(width, dtype=np.float32)[None, :]
        top_arr = np.asarray(top, dtype=np.float32)[None, None, :]
        bottom_arr = np.asarray(bottom, dtype=np.float32)[None, None, :]
        data = top_arr * (1.0 - ys[..., None]) + bottom_arr * ys[..., None]
        wave = 2.0 * np.sin((xs / max(1, width)) * math.pi * 2.0)[..., None]
        data = np.clip(data + wave, 0, 255).astype(np.uint8)
        image = Image.fromarray(data, mode="RGB")
    else:
        image = Image.new("RGB", (width, height))
        pixels = image.load()
        for y in range(height):
            t = y / max(1, height - 1)
            row = tuple(_lerp(top[i], bottom[i], t) for i in range(3))
            for x in range(width):
                # A very subtle deterministic horizontal modulation gives the
                # feature matcher something to follow without adding random noise.
                wave = 2.0 * math.sin((x / width) * math.pi * 2.0)
                pixels[x, y] = tuple(_clamp(channel + wave) for channel in row)
    draw = ImageDraw.Draw(image)
    if changed:
        draw.rectangle((0, int(height * 0.73), width, height), fill=(18, 84, 82))
        for x in range(0, width, 64):
            draw.line((x, int(height * 0.73), x + 24, height), fill=(31, 111, 105), width=2)
    else:
        draw.rectangle((0, int(height * 0.77), width, height), fill=(18, 30, 51))
        for x in range(-height, width, 84):
            draw.line((x, height, x + height, int(height * 0.77)), fill=(28, 44, 68), width=2)
    # Fixed stars / lights make the background-change evidence visible.
    for x, y, r in ((72, 54, 2), (176, 80, 1), (524, 48, 2), (582, 108, 1), (398, 34, 1)):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(205, 220, 255))
    return image


def _apply_brightness(image: Image.Image, factor: float) -> Image.Image:
    if abs(factor - 1.0) < 1e-9:
        return image
    # Keep the dependency surface tiny; Pillow's point operation is stable.
    lut = [_clamp(i * factor) for i in range(256)]
    return image.convert("RGB").point(lut * 3)


def _draw_character(
    image: Image.Image,
    *,
    frame_number: int,
    scale: float,
    costume_color: tuple[int, int, int],
) -> dict[str, Any]:
    """Draw the stable character and return its reference bounding boxes."""

    draw = ImageDraw.Draw(image)
    width, height = image.size
    # A small, smooth walk cycle makes optical-flow and adjacent comparisons
    # meaningful while leaving the intentional scale jump unambiguous.
    cx = width * 0.50 + math.sin(frame_number * 0.35) * 18.0
    ground = height * 0.80
    s = scale
    body_w, body_h = 76 * s, 112 * s
    head_r = 34 * s
    head_cy = ground - body_h - head_r * 0.78
    torso_top = head_cy + head_r * 0.70
    torso_bottom = torso_top + body_h * 0.62
    left = cx - body_w / 2
    right = cx + body_w / 2

    # Hair silhouette (the analysis treats this as a proxy for a hair mask).
    hair_box = (cx - head_r * 1.18, head_cy - head_r * 1.18, cx + head_r * 1.18, head_cy + head_r * 1.12)
    draw.ellipse(hair_box, fill=(40, 25, 65), outline=(117, 86, 163), width=max(1, int(2 * s)))
    face_box = (cx - head_r * 0.86, head_cy - head_r * 0.82, cx + head_r * 0.86, head_cy + head_r * 0.90)
    draw.ellipse(face_box, fill=(242, 195, 167), outline=(215, 157, 130), width=max(1, int(1 * s)))
    # Hair fringe / side locks.
    fringe = [(cx - head_r, head_cy - head_r * 0.20), (cx - head_r * 0.35, head_cy - head_r * 0.94),
              (cx, head_cy - head_r * 0.67), (cx + head_r * 0.38, head_cy - head_r * 0.93),
              (cx + head_r, head_cy - head_r * 0.16), (cx + head_r * 0.62, head_cy + head_r * 0.08),
              (cx, head_cy - head_r * 0.13), (cx - head_r * 0.62, head_cy + head_r * 0.08)]
    draw.polygon(fringe, fill=(51, 32, 79))
    # Eyes.
    eye_y = head_cy + head_r * 0.18
    for eye_x in (cx - head_r * 0.34, cx + head_r * 0.34):
        draw.ellipse((eye_x - 3 * s, eye_y - 3 * s, eye_x + 3 * s, eye_y + 3 * s), fill=(38, 48, 85))
    # Costume / torso.
    draw.rounded_rectangle((left, torso_top, right, torso_bottom), radius=max(2, int(8 * s)), fill=costume_color, outline=(220, 230, 240), width=max(1, int(2 * s)))
    draw.line((cx, torso_top + 8 * s, cx, torso_bottom - 6 * s), fill=(235, 235, 245), width=max(1, int(2 * s)))
    # Arms, legs, and shoes.
    arm_y = torso_top + 16 * s
    draw.line((left + 5 * s, arm_y, cx - body_w * 0.72, torso_bottom - 2 * s), fill=(242, 195, 167), width=max(2, int(10 * s)))
    draw.line((right - 5 * s, arm_y, cx + body_w * 0.72, torso_bottom - 2 * s), fill=(242, 195, 167), width=max(2, int(10 * s)))
    hip_y = torso_bottom
    draw.line((cx - body_w * 0.23, hip_y, cx - body_w * 0.30, ground - 3 * s), fill=(53, 66, 100), width=max(3, int(14 * s)))
    draw.line((cx + body_w * 0.23, hip_y, cx + body_w * 0.30, ground - 3 * s), fill=(53, 66, 100), width=max(3, int(14 * s)))
    draw.line((cx - body_w * 0.34, ground - 2 * s, cx - body_w * 0.52, ground), fill=(26, 28, 38), width=max(3, int(16 * s)))
    draw.line((cx + body_w * 0.34, ground - 2 * s, cx + body_w * 0.52, ground), fill=(26, 28, 38), width=max(3, int(16 * s)))

    # Return boxes in image coordinates.  These are included in the manifest so
    # a consumer can lock an ROI without needing a detector.
    character_box = (
        int(max(0, cx - head_r * 1.35)),
        int(max(0, head_cy - head_r * 1.35)),
        int(min(width, cx + head_r * 1.35)),
        int(min(height, ground + 2 * s)),
    )
    face = tuple(int(v) for v in face_box)
    hair = tuple(int(v) for v in hair_box)
    costume = (int(max(0, left)), int(max(0, torso_top)), int(min(width, right)), int(min(height, torso_bottom)))
    return {"character": character_box, "face": face, "hair": hair, "costume": costume, "center": [round(cx, 2), round((head_cy + ground) / 2, 2)], "scale": round(scale, 3)}


def frame_spec(frame_number: int) -> dict[str, Any]:
    """Return the deterministic perturbations for one frame."""

    # Deliberately explicit ranges: tests and users can inspect these markers.
    color_drift = frame_number in (8, 9, 10)
    scale_jump = frame_number == 13
    brightness_flicker = frame_number in (16, 18)
    background_change = frame_number in (20, 21, 22)
    return {
        "frame_number": frame_number,
        "color_drift": color_drift,
        "scale_jump": scale_jump,
        "brightness_flicker": brightness_flicker,
        "background_change": background_change,
        "scale": 1.34 if scale_jump else 1.0,
        "brightness_factor": (1.30 if frame_number == 16 else 0.68 if frame_number == 18 else 1.0),
        "costume_color": [193, 78, 106] if color_drift else [63, 138, 190],
        "background_variant": "changed" if background_change else "base",
    }


def _write_preview_montage(frames_dir: Path, output_path: Path, frame_count: int) -> None:
    """Write a compact visual key for the demo without any generated imagery."""

    interesting = [number for number in (1, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23) if number <= frame_count]
    thumbnails: list[tuple[int, Image.Image]] = []
    for number in interesting:
        path = frames_dir / f"frame_{number:04d}.png"
        if not path.exists():
            continue
        with Image.open(path) as source:
            thumbnails.append((number, source.convert("RGB").resize((240, 135))))
    if not thumbnails:
        return
    columns, cell_width, cell_height = 4, 240, 158
    rows = math.ceil(len(thumbnails) / columns)
    montage = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(montage)
    for index, (number, thumb) in enumerate(thumbnails):
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        montage.paste(thumb, (x, y))
        draw.text((x + 5, y + 138), f"Frame {number:02d}", fill="#18243a")
    montage.save(output_path, format="PNG", optimize=True)


def generate_mock_sequence(
    output_dir: str | Path,
    *,
    frame_count: int = DEFAULT_FRAMES,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: float = DEFAULT_FPS,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Generate PNG frames and return the written manifest as a dictionary."""

    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if width < 8 or height < 8:
        raise ValueError("width and height must be at least 8 pixels")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for frame_number in range(1, frame_count + 1):
        spec = frame_spec(frame_number)
        image = _gradient_background(width, height, changed=spec["background_change"])
        boxes = _draw_character(
            image,
            frame_number=frame_number,
            scale=spec["scale"],
            costume_color=tuple(spec["costume_color"]),
        )
        image = _apply_brightness(image, spec["brightness_factor"])
        filename = f"frame_{frame_number:04d}.png"
        path = frames_dir / filename
        if overwrite or not path.exists():
            image.save(path, format="PNG", optimize=False)
        entries.append(
            {
                "frame_id": f"f{frame_number:04d}",
                "frame_number": frame_number,
                "timestamp": round((frame_number - 1) / fps, 6),
                "file": str(Path("frames") / filename),
                "shot": "demo-shot-001",
                "character": "demo-character",
                "status": "generated",
                "roi": boxes,
                "markers": {
                    "color_drift": spec["color_drift"],
                    "scale_jump": spec["scale_jump"],
                    "brightness_flicker": spec["brightness_flicker"],
                    "background_change": spec["background_change"],
                },
            }
        )
    color_frames = [number for number in (8, 9, 10) if number <= frame_count]
    scale_frames = [number for number in (13,) if number <= frame_count]
    flicker_frames = [number for number in (16, 18) if number <= frame_count]
    background_frames = [number for number in (20, 21, 22) if number <= frame_count]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sequence_id": "mock-continuity-sequence-001",
        "name": "Visual Continuity Lab deterministic demo",
        "source": "procedural-local",
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "master_reference": "frames/frame_0001.png",
        "intentional_events": {
            "color_drift": {"frames": color_frames, "description": "Costume shifts from blue to magenta."},
            "scale_jump": {"frames": scale_frames, "description": "Character scale is 1.34x for one frame."},
            "brightness_flicker": {"frames": flicker_frames, "description": "Alternating bright and dark exposure."},
            "background_change": {"frames": background_frames, "description": "Ground and sky palette changes."},
        },
        # A compact alias is convenient for importers that only need to know
        # which frame numbers should produce a non-low drift result.
        "demo_markers": {
            "color_drift": color_frames,
            "scale_jump": scale_frames,
            "brightness_flicker": flicker_frames,
            "background_change": background_frames,
        },
        "frames": entries,
    }
    manifest_path = output / "sequence.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_preview_montage(frames_dir, output / "preview_montage.png", frame_count)
    # A short human-readable marker file is handy when browsing the demo folder.
    (output / "README.txt").write_text(
        "Deterministic Visual Continuity Lab demo.\n"
        "preview_montage.png (when present) shows the planted continuity defects.\n"
        "Run: python scripts/analyze_demo.py analyze --input demo/mock_sequence --output artifacts/demo_analysis\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="demo/mock_sequence", help="output directory")
    parser.add_argument("--output", dest="output_option", help="output directory (named form)")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, dest="frame_count")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    args = parser.parse_args(argv)
    output = args.output_option or args.output
    manifest = generate_mock_sequence(output, frame_count=args.frame_count, width=args.width, height=args.height, fps=args.fps)
    print(json.dumps({"output": str(Path(output)), "frame_count": manifest["frame_count"], "events": manifest["intentional_events"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
