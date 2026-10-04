"""Input adapters for still images, folders, frame sequences and videos."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Any

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

from .models import FrameRecord, ROI, Sequence


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"
}


def _natural_key(path: Path):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def _normalise_path(source: str | Path) -> Path:
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_image(source: str | Path | Any):
    """Load an image as RGB Pillow image (or an RGB NumPy array fallback)."""

    if Image is not None and isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (str, Path)):
        path = _normalise_path(source)
        if Image is None:
            if cv2 is None:
                raise RuntimeError("Pillow or OpenCV is required to load images")
            data = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if data is None:
                raise ValueError(f"Unable to read image: {path}")
            return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        with Image.open(path) as opened:
            # Copy before closing the context manager so callers can retain it.
            return opened.convert("RGB").copy()
    if np is not None:
        arr = np.asarray(source)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        return arr
    return source


def _frame_id(path: Path, index: int) -> str:
    return path.stem or f"frame_{index:06d}"


def sequence_from_paths(
    paths: Iterable[str | Path],
    *,
    sequence_id: str | None = None,
    shot: str | None = None,
    character: str | None = None,
    start_frame: int = 1,
    fps: float | None = None,
) -> Sequence:
    """Build a :class:`Sequence` from image paths in deterministic order."""

    path_list = [_normalise_path(p) for p in paths]
    path_list = [p for p in path_list if p.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS]
    path_list.sort(key=_natural_key)
    sid = sequence_id or (path_list[0].parent.name if path_list else "sequence")
    frames = []
    for i, path in enumerate(path_list):
        number = int(start_frame) + i
        frames.append(FrameRecord(
            frame_id=_frame_id(path, i), path=str(path), frame_number=number,
            timestamp=(number - start_frame) / float(fps) if fps and fps > 0 else None,
            shot=shot, character=character,
        ))
    return Sequence(sequence_id=sid, frames=frames, shot=shot, character=character,
                    metadata={"source_type": "images", "fps": fps})


def sequence_from_folder(
    folder: str | Path,
    *,
    sequence_id: str | None = None,
    recursive: bool = False,
    shot: str | None = None,
    character: str | None = None,
    start_frame: int = 1,
    fps: float | None = None,
) -> Sequence:
    """Load all supported images from ``folder`` (optionally recursively)."""

    root = _normalise_path(folder)
    if not root.is_dir():
        raise NotADirectoryError(root)
    iterator = root.rglob("*") if recursive else root.glob("*")
    result = sequence_from_paths(iterator, sequence_id=sequence_id or root.name, shot=shot,
                                 character=character, start_frame=start_frame, fps=fps)
    # If the folder is part of a generated/imported sequence, honor its local
    # manifest ROI.  This is especially valuable for stylized characters where
    # a generic foreground detector would include the gradient background.
    manifest_path = root / "sequence.json"
    if not manifest_path.exists() and root.parent.joinpath("sequence.json").exists():
        manifest_path = root.parent / "sequence.json"
    if manifest_path.exists():
        try:
            import json
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = manifest.get("frames", [])
            by_name = {Path(str(item.get("file", ""))).name: item for item in entries if isinstance(item, dict)}
            by_id = {str(item.get("frame_id")): item for item in entries if isinstance(item, dict) and item.get("frame_id")}
            for frame in result.frames:
                item = by_name.get(Path(frame.path or "").name, {})
                if not item:
                    item = by_id.get(frame.frame_id, {})
                if not item:
                    continue
                frame.frame_id = str(item.get("frame_id", frame.frame_id))
                frame.frame_number = int(item.get("frame_number", frame.frame_number) or frame.frame_number)
                frame.timestamp = float(item["timestamp"]) if item.get("timestamp") is not None else frame.timestamp
                frame.shot = item.get("shot", frame.shot)
                frame.character = item.get("character", frame.character)
                frame.metadata.update({"manifest": item, "markers": item.get("markers", {})})
                bbox = (item.get("roi") or {}).get("character")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    # Manifest dimensions are authoritative for pixel boxes.
                    width = float(manifest.get("width") or 1)
                    height = float(manifest.get("height") or 1)
                    x1, y1, x2, y2 = (float(v) for v in bbox)
                    frame.roi = ROI(
                        x1 / width, y1 / height, max(1.0, x2 - x1) / width,
                        max(1.0, y2 - y1) / height, True, True, "manifest-character"
                    ).clamp()
            master = manifest.get("master_reference")
            if master:
                master_name = Path(str(master)).name
                match = next((f for f in result.frames if Path(f.path or "").name == master_name), None)
                result.master_reference = match.frame_id if match else str(master)
            result.sequence_id = str(manifest.get("sequence_id", result.sequence_id))
            result.shot = shot or result.shot
            result.character = character or result.character
            result.metadata.update({"manifest_path": str(manifest_path), "manifest": manifest})
        except Exception:
            # A malformed optional manifest must not prevent basic folder use.
            pass
    return result


def sequence_from_image(
    image: str | Path,
    *,
    sequence_id: str | None = None,
    shot: str | None = None,
    character: str | None = None,
) -> Sequence:
    path = _normalise_path(image)
    return sequence_from_paths([path], sequence_id=sequence_id or path.stem,
                               shot=shot, character=character)


def sequence_from_video(
    video: str | Path,
    *,
    output_dir: str | Path | None = None,
    every_n: int = 1,
    max_frames: int | None = None,
    sequence_id: str | None = None,
    shot: str | None = None,
    character: str | None = None,
) -> Sequence:
    """Extract a video into PNG frames and return a :class:`Sequence`.

    Extracted files are kept on disk so report JSON remains useful after the
    analyzer exits.  If no output directory is supplied a temporary directory
    is created and recorded in sequence metadata.
    """

    path = _normalise_path(video)
    if cv2 is None:
        raise RuntimeError("Video extraction requires OpenCV (cv2)")
    every_n = max(1, int(every_n))
    target = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="vclab_frames_"))
    target.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[FrameRecord] = []
    index = 0
    saved = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if index % every_n == 0:
                file = target / f"frame_{index:06d}.png"
                if not cv2.imwrite(str(file), bgr):
                    raise IOError(f"Unable to write extracted frame: {file}")
                frames.append(FrameRecord(
                    frame_id=file.stem, path=str(file), frame_number=index + 1,
                    timestamp=(index / fps if fps > 0 else None), shot=shot,
                    character=character,
                ))
                saved += 1
                if max_frames is not None and saved >= int(max_frames):
                    break
            index += 1
    finally:
        cap.release()
    return Sequence(
        sequence_id=sequence_id or path.stem,
        frames=frames,
        shot=shot,
        character=character,
        metadata={"source_type": "video", "source": str(path), "output_dir": str(target),
                  "fps": fps, "every_n": every_n},
    )


def sequence_from_source(source: str | Path, **kwargs) -> Sequence:
    """Dispatch a path to image, folder, or video loading."""

    path = _normalise_path(source)
    if path.is_dir():
        return sequence_from_folder(path, **kwargs)
    if path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS:
        return sequence_from_image(path, **kwargs)
    if path.suffix.casefold() == ".json":
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "shots" in payload:
            from .io import sequence_from_frameforge
            result = sequence_from_frameforge(payload)
        else:
            result = Sequence.from_dict(payload)
        # Convert manifest-style named pixel boxes into the canonical normalized
        # character ROI when dimensions are available.
        manifest_w = float(payload.get("width") or payload.get("sourceWidth") or 0)
        manifest_h = float(payload.get("height") or payload.get("sourceHeight") or 0)
        if manifest_w > 0 and manifest_h > 0:
            for frame in result.frames:
                boxes = frame.metadata.get("roi_boxes") if isinstance(frame.metadata, dict) else None
                bbox = boxes.get("character") if isinstance(boxes, dict) else None
                if frame.roi is None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x1, y1, x2, y2 = (float(v) for v in bbox)
                    frame.roi = ROI(x1 / manifest_w, y1 / manifest_h,
                                    max(1.0, x2 - x1) / manifest_w,
                                    max(1.0, y2 - y1) / manifest_h,
                                    True, True, "manifest-character").clamp()
        # Resolve relative frame paths against the manifest directory.
        for frame in result.frames:
            if frame.path and not Path(frame.path).is_absolute():
                candidate = path.parent / frame.path
                if candidate.exists():
                    frame.path = str(candidate)
        return result
    return sequence_from_video(path, **kwargs)


def sequence_from_shot(shot: dict[str, Any] | Sequence, *, sequence_id: str | None = None, **kwargs: Any) -> Sequence:
    """Create a sequence from a FrameForge-style shot dictionary."""
    if isinstance(shot, Sequence):
        return shot
    payload = dict(shot)
    payload.setdefault("sequence_id", sequence_id or payload.get("shotId", "shot"))
    payload.setdefault("frames", payload.get("frames", []))
    payload.setdefault("shot", payload.get("shotId"))
    payload.update(kwargs)
    return Sequence.from_dict(payload)


__all__ = [
    "FrameSequence", "SUPPORTED_IMAGE_EXTENSIONS", "load_image", "sequence_from_folder",
    "sequence_from_image", "sequence_from_paths", "sequence_from_source",
    "sequence_from_shot", "sequence_from_video",
]

# Friendly aliases used by some integrations.
FrameSequence = Sequence
