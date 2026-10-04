"""UI-facing adapter for the continuity analysis engine.

The desktop UI must remain usable while the analysis engine is being developed.
This module therefore exposes a deliberately boring, dictionary based contract
and attempts to call a core pipeline when one is available.  If no compatible
core is installed, a local deterministic analyser is used.  The fallback is
not a replacement for a production detector; it is a useful smoke-test and a
real baseline for image sequences.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass
class FrameRecord:
    frame_id: str
    frame_number: int
    path: str
    timestamp: float = 0.0
    character: str = "Character 1"
    shot: str = "Shot 1"
    status: str = "Pending"
    # Optional manifest/ROI metadata is kept on the UI-side record so a
    # generated sequence can be analyzed consistently without requiring the
    # user to redraw a region every time it is opened.  The fields are
    # intentionally permissive dictionaries; the core adapter converts them
    # to its typed ROI model.
    roi: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _natural_key(path: Path) -> tuple[Any, ...]:
    import re

    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name))


def _image_paths(paths: Iterable[str | os.PathLike[str]]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTENSIONS)
        elif p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    # Keep first occurrence when a folder and an image were both selected.
    unique = {str(x.resolve()): x for x in files if x.exists()}
    return sorted(unique.values(), key=_natural_key)


class AnalysisAdapter:
    """Bridge between the UI and the optional ``vclab.core`` pipeline.

    Core integration options, in order:

    * ``vclab.core.pipeline.analyze_sequence(records, roi=..., master=...)``
    * ``vclab.core.analyzer.analyze_sequence(...)``
    * an object exposing ``analyze_sequence`` in either module.

    Returned values are normalized to ``{health, drift, issues, frames}``.
    """

    def __init__(self) -> None:
        self._core = self._discover_core()
        self._video_tempdir = tempfile.TemporaryDirectory(prefix="vclab-video-")

    @staticmethod
    def _discover_core() -> Any | None:
        candidates = ("vclab.core.pipeline", "vclab.core.analyzer", "vclab.core")
        for module_name in candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            if callable(getattr(module, "analyze_sequence", None)):
                return module
            for attr in ("ContinuityAnalyzer", "Analyzer", "Pipeline"):
                cls = getattr(module, attr, None)
                if cls is not None:
                    try:
                        obj = cls()
                    except Exception:
                        continue
                    if callable(getattr(obj, "analyze_sequence", None)):
                        return obj
        return None

    @property
    def core_available(self) -> bool:
        return self._core is not None

    def load_paths(self, paths: Iterable[str | os.PathLike[str]]) -> list[FrameRecord]:
        """Build records from image files/folders and extract selected videos."""
        records: list[FrameRecord] = []
        frame_no = 1
        for raw in paths:
            p = Path(raw)
            if p.is_dir() or p.suffix.lower() in IMAGE_EXTENSIONS:
                for image_path in _image_paths([p]):
                    records.append(
                        FrameRecord(
                            frame_id=f"F{frame_no:04d}",
                            frame_number=frame_no,
                            path=str(image_path),
                            timestamp=0.0,
                        )
                    )
                    frame_no += 1
            elif p.suffix.lower() in VIDEO_EXTENSIONS:
                records.extend(self._extract_video(p, frame_no))
                frame_no = len(records) + 1
        # Re-number after mixing sources while retaining a stable frame id.
        for i, record in enumerate(records, 1):
            record.frame_number = i
            record.frame_id = f"F{i:04d}"
        self._apply_manifest_rois(records)
        return records

    @staticmethod
    def _apply_manifest_rois(records: Sequence[FrameRecord]) -> None:
        """Attach optional generated-sequence ROI metadata to UI records.

        A folder may contain ``sequence.json`` next to ``frames/``.  Reading
        this small, local manifest is not a FrameForge dependency; it simply
        preserves a previously locked character region and stable timestamps.
        Malformed or absent manifests are ignored so ordinary image folders
        retain the normal draw-an-ROI workflow.
        """
        manifests: dict[str, dict[str, Any]] = {}
        for record in records:
            path = Path(record.path)
            candidates = [path.parent / "sequence.json", path.parent.parent / "sequence.json"]
            for manifest_path in candidates:
                if not manifest_path.exists():
                    continue
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    width = float(data.get("width") or 0.0)
                    height = float(data.get("height") or 0.0)
                    for item in data.get("frames", []):
                        if not isinstance(item, dict):
                            continue
                        filename = Path(str(item.get("file", ""))).name
                        if filename != path.name:
                            continue
                        bbox = (item.get("roi") or {}).get("character")
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and width > 0 and height > 0:
                            x1, y1, x2, y2 = (float(v) for v in bbox)
                            item_roi = {
                                "x": x1 / width,
                                "y": y1 / height,
                                "width": max(1.0, x2 - x1) / width,
                                "height": max(1.0, y2 - y1) / height,
                                "normalized": True,
                                "locked": True,
                                "label": "manifest-character",
                            }
                            record.roi = item_roi
                            record.metadata.update({"manifest": item, "markers": item.get("markers", {})})
                            if item.get("timestamp") is not None:
                                record.timestamp = float(item["timestamp"])
                        break
                except Exception:
                    continue

    def _extract_video(self, path: Path, start: int) -> list[FrameRecord]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return []
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out: list[FrameRecord] = []
        # Keep every frame for short clips; sample long clips so loading remains responsive.
        stride = max(1, math.ceil(count / 240)) if count else 1
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                target = Path(self._video_tempdir.name) / f"{path.stem}_{index:06d}.png"
                cv2.imwrite(str(target), frame)
                out.append(
                    FrameRecord(
                        frame_id=f"F{start + len(out):04d}",
                        frame_number=start + len(out),
                        path=str(target),
                        timestamp=index / fps,
                        shot=path.stem,
                    )
                )
            index += 1
        capture.release()
        return out

    def analyze(
        self,
        records: Sequence[FrameRecord],
        roi: tuple[float, float, float, float] | None = None,
        master_index: int | None = None,
    ) -> dict[str, Any]:
        if not records:
            return {"health": {}, "drift": [], "issues": [], "frames": []}
        if self._core is not None:
            try:
                raw = self._call_core(records, roi=roi, master_index=master_index)
                if raw is not None:
                    return self._normalize_result(raw, records)
            except Exception:
                # The fallback keeps the UI functional if a partially implemented
                # core changes its signature during development.
                pass
        return self._fallback_analyze(records, roi=roi, master_index=master_index)

    def _call_core(
        self,
        records: Sequence[FrameRecord],
        roi: tuple[float, float, float, float] | None,
        master_index: int | None,
    ) -> Any | None:
        """Call a core analyzer across the supported development signatures."""
        payload = [r.to_dict() for r in records]
        # Prefer the canonical model API when available.  Importing lazily keeps
        # the UI adapter usable while the engine package is being assembled.
        sequence_obj: Any = payload
        try:
            models = importlib.import_module("vclab.core.models")
            sequence_cls = getattr(models, "Sequence", None)
            frame_cls = getattr(models, "FrameRecord", None)
            roi_cls = getattr(models, "ROI", None)
            if sequence_cls and frame_cls:
                frames = [frame_cls.from_dict(item) if hasattr(frame_cls, "from_dict") else frame_cls(**item) for item in payload]
                sequence_obj = sequence_cls("ui-sequence", frames=frames)
                if roi and roi_cls and hasattr(sequence_obj, "set_roi"):
                    sequence_obj.set_roi(roi_cls(*roi, normalized=True), lock=True)
        except Exception:
            sequence_obj = payload
        fn = getattr(self._core, "analyze_sequence", None)
        if callable(fn):
            # Signature variants used by early engine iterations.
            for args, kwargs in (
                ((sequence_obj,), {"master_index": master_index or 0}),
                ((sequence_obj,), {"master": master_index or 0, "roi": roi}),
                ((payload,), {"master_index": master_index or 0}),
                ((payload,), {"master": master_index or 0, "roi": roi}),
            ):
                try:
                    return fn(*args, **kwargs)
                except (TypeError, AttributeError):
                    continue
        analyzer = self._core
        method = getattr(analyzer, "analyze", None)
        if callable(method):
            for args, kwargs in (
                ((sequence_obj,), {"master_index": master_index or 0}),
                ((payload,), {"master_index": master_index or 0}),
            ):
                try:
                    return method(*args, **kwargs)
                except (TypeError, AttributeError):
                    continue
        return None

    @staticmethod
    def _normalize_result(raw: Any, records: Sequence[FrameRecord]) -> dict[str, Any]:
        if hasattr(raw, "to_dict"):
            try:
                raw = raw.to_dict()
            except TypeError:
                raw = raw.to_dict(include_features=False)
        if not isinstance(raw, dict):
            return AnalysisAdapter._fallback_analyze(records)
        raw_health = raw.get("health", {})
        if hasattr(raw_health, "to_dict"):
            raw_health = raw_health.to_dict()
        raw_health = dict(raw_health or {})
        health_names = {
            "identity_stability": "Identity Stability",
            "motion_stability": "Motion Stability",
            "color_stability": "Color Stability",
            "lighting_stability": "Lighting Stability",
            "background_stability": "Background Stability",
            "overall": "Overall",
        }
        health = {health_names.get(key, key): value for key, value in raw_health.items()}
        raw_drift = list(raw.get("drift", raw.get("transitions", raw.get("adjacent_results", []))) or [])
        drift: list[dict[str, Any]] = []
        master_drift: list[dict[str, Any]] = []
        frame_lookup = {r.frame_id: i + 1 for i, r in enumerate(records)}
        dimension_aliases = {
            "face_region": "Face Drift",
            "hair_silhouette": "Hair Drift",
            "hair_color": "Hair Color Drift",
            "costume_color": "Color Drift",
            "body_proportion": "Body Proportion Drift",
            "character_position": "Character Position Drift",
            "character_scale": "Character Scale Jump",
            "camera_composition": "Camera Composition",
            "background": "Background Drift",
            "lighting": "Brightness Flicker",
            "sharpness": "Sharpness Drift",
        }
        for item in raw_drift:
            if hasattr(item, "to_dict"):
                item = item.to_dict()
            item = dict(item or {})
            dimensions: dict[str, dict[str, Any]] = {}
            for name, dim in dict(item.get("dimensions", {})).items():
                if hasattr(dim, "to_dict"):
                    dim = dim.to_dict()
                dim = dict(dim or {})
                dimensions[dimension_aliases.get(name, name)] = {
                    "score": float(dim.get("score", 0.0)),
                    "level": str(dim.get("severity", dim.get("level", "Low"))).title(),
                    "evidence": dim.get("evidence", {}),
                }
            from_id = item.get("from_frame", item.get("frame_from", ""))
            to_id = item.get("to_frame", item.get("frame_to", ""))
            from_no = frame_lookup.get(str(from_id), from_id)
            to_no = frame_lookup.get(str(to_id), to_id)
            drift.append(
                {
                    "from_frame": from_no,
                    "to_frame": to_no,
                    "frame": to_no,
                    "score": float(item.get("overall_score", item.get("score", 0.0))),
                    "master_score": float(item.get("master_score", 0.0)),
                    "dimensions": dimensions,
                }
            )
        # Master-reference comparisons use the same compact row shape.  Keeping
        # them separate lets the UI/report consumers distinguish adjacent drift
        # from cumulative drift against the locked master frame.
        for item in list(raw.get("master_results", []) or []):
            if hasattr(item, "to_dict"):
                item = item.to_dict()
            item = dict(item or {})
            dimensions: dict[str, dict[str, Any]] = {}
            for name, dim in dict(item.get("dimensions", {})).items():
                if hasattr(dim, "to_dict"):
                    dim = dim.to_dict()
                dim = dict(dim or {})
                dimensions[dimension_aliases.get(name, name)] = {
                    "score": float(dim.get("score", 0.0)),
                    "level": str(dim.get("severity", dim.get("level", "Low"))).title(),
                    "evidence": dim.get("evidence", {}),
                }
            from_id = item.get("from_frame", item.get("frame_from", ""))
            to_id = item.get("to_frame", item.get("frame_to", ""))
            master_drift.append(
                {
                    "from_frame": frame_lookup.get(str(from_id), from_id),
                    "to_frame": frame_lookup.get(str(to_id), to_id),
                    "frame": frame_lookup.get(str(to_id), to_id),
                    "score": float(item.get("overall_score", item.get("score", 0.0))),
                    "dimensions": dimensions,
                }
            )
        issues: list[dict[str, Any]] = []
        for item in list(raw.get("issues", []) or []):
            if hasattr(item, "to_dict"):
                item = item.to_dict()
            item = dict(item or {})
            evidence = item.get("evidence", {})
            if isinstance(evidence, dict):
                evidence_text = "; ".join(f"{k}: {v}" for k, v in evidence.items())
            else:
                evidence_text = str(evidence)
            issues.append(
                {
                    "type": item.get("type", "Unknown"),
                    "start_frame": frame_lookup.get(str(item.get("start_frame", "")), item.get("start_frame", "—")),
                    "end_frame": frame_lookup.get(str(item.get("end_frame", "")), item.get("end_frame", "—")),
                    "severity": str(item.get("severity", "Low")).title(),
                    "evidence": evidence_text,
                    "notes": item.get("notes", ""),
                }
            )
        frames = list(raw.get("frames", raw.get("frame_metrics", [])) or [])
        if not frames and raw.get("frame_features"):
            for frame_id, feature in raw["frame_features"].items():
                if hasattr(feature, "to_dict"):
                    feature = feature.to_dict()
                feature = dict(feature or {})
                frames.append({"frame_id": frame_id, **feature})
        return {
            "health": health,
            "drift": drift,
            "master_drift": master_drift,
            "issues": issues,
            "frames": frames,
            "bad_frames": list(raw.get("bad_frames", []) or []),
            "recommended_frames": list(raw.get("recommended_frames", []) or []),
        }

    @staticmethod
    def _open_rgb(path: str) -> Image.Image:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    @staticmethod
    def _crop_roi(image: Image.Image, roi: tuple[float, float, float, float] | None) -> Image.Image:
        if not roi:
            return image
        x, y, w, h = roi
        x0 = max(0, min(image.width - 1, int(x * image.width)))
        y0 = max(0, min(image.height - 1, int(y * image.height)))
        x1 = max(x0 + 1, min(image.width, int((x + w) * image.width)))
        y1 = max(y0 + 1, min(image.height, int((y + h) * image.height)))
        return image.crop((x0, y0, x1, y1))

    @staticmethod
    def _features(image: Image.Image, roi: tuple[float, float, float, float] | None) -> dict[str, Any]:
        image = AnalysisAdapter._crop_roi(image, roi)
        arr = np.asarray(image, dtype=np.float32)
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        # A compact perceptual hash representation for deterministic comparisons.
        phash = (small > float(small.mean())).astype(np.uint8).reshape(-1)
        edges = cv2.Canny(gray, 80, 160)
        return {
            "width": image.width,
            "height": image.height,
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "color": arr.mean(axis=(0, 1)).tolist(),
            "phash": phash,
            "edge_density": float((edges > 0).mean()),
        }

    @staticmethod
    def _similarity(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
        brightness = min(1.0, abs(a["brightness"] - b["brightness"]) / 64.0)
        contrast = min(1.0, abs(a["contrast"] - b["contrast"]) / 48.0)
        color = min(1.0, float(np.linalg.norm(np.asarray(a["color"]) - np.asarray(b["color"])) / 160.0))
        phash = float(np.count_nonzero(a["phash"] != b["phash"]) / len(a["phash"]))
        edge = min(1.0, abs(a["edge_density"] - b["edge_density"]) / 0.25)
        return {
            "color": color,
            "brightness": brightness,
            "contrast": contrast,
            "background": edge,
            "identity": min(1.0, (phash * 0.7 + edge * 0.3)),
        }

    @staticmethod
    def _level(value: float) -> str:
        if value >= 0.7:
            return "High"
        if value >= 0.35:
            return "Medium"
        return "Low"

    def _fallback_analyze(
        self,
        records: Sequence[FrameRecord],
        roi: tuple[float, float, float, float] | None = None,
        master_index: int | None = None,
    ) -> dict[str, Any]:
        feature_rows: list[dict[str, Any]] = []
        for record in records:
            try:
                image = self._open_rgb(record.path)
                f = self._features(image, roi)
            except Exception:
                f = {"brightness": 0.0, "contrast": 0.0, "color": [0, 0, 0], "phash": np.zeros(1024, dtype=np.uint8), "edge_density": 0.0, "width": 0, "height": 0}
            feature_rows.append(f)
        master_position = max(0, min(len(feature_rows) - 1, int(master_index or 0)))
        master = feature_rows[master_position]
        drift: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        dimension_keys = {
            "Color Drift": "color",
            "Brightness Flicker": "brightness",
            "Background Drift": "background",
            "Identity Drift": "identity",
        }
        for i in range(1, len(records)):
            metrics = self._similarity(feature_rows[i - 1], feature_rows[i])
            master_metrics = self._similarity(master, feature_rows[i])
            row = {
                "from_frame": i,
                "to_frame": i + 1,
                "frame": i + 1,
                "score": round(float(np.mean(list(metrics.values()))), 4),
                "master_score": round(float(np.mean(list(master_metrics.values()))), 4),
                "dimensions": {
                    name: {"score": round(metrics[key], 4), "level": self._level(metrics[key])}
                    for name, key in dimension_keys.items()
                },
            }
            # Proxy dimensions are still useful in a baseline: edge/identity drift
            # approximates silhouette and position changes when no detector exists.
            row["dimensions"].update(
                {
                    "Hair Drift": {"score": round(metrics["identity"] * 0.9, 4), "level": self._level(metrics["identity"] * 0.9)},
                    "Character Scale Jump": {"score": round(metrics["background"] * 0.75, 4), "level": self._level(metrics["background"] * 0.75)},
                    "Camera Composition": {"score": round(metrics["edge_density"] if "edge_density" in metrics else metrics["background"], 4), "level": self._level(metrics["background"])},
                }
            )
            drift.append(row)
            for label, payload in row["dimensions"].items():
                if payload["score"] >= 0.7:
                    issues.append(
                        {
                            "type": label,
                            "start_frame": i,
                            "end_frame": i + 1,
                            "severity": "High",
                            "evidence": f"{label} score {payload['score']:.2f} between F{i:04d} and F{i+1:04d}",
                            "notes": "Review transition and consider regenerating the later frame.",
                        }
                    )
        master_drift: list[dict[str, Any]] = []
        for i in range(1, len(records)):
            metrics = self._similarity(master, feature_rows[i])
            dimensions = {
                name: {"score": round(metrics[key], 4), "level": self._level(metrics[key])}
                for name, key in dimension_keys.items()
            }
            dimensions.update(
                {
                    "Hair Drift": {"score": round(metrics["identity"] * 0.9, 4), "level": self._level(metrics["identity"] * 0.9)},
                    "Character Scale Jump": {"score": round(metrics["background"] * 0.75, 4), "level": self._level(metrics["background"] * 0.75)},
                    "Camera Composition": {"score": round(metrics["background"], 4), "level": self._level(metrics["background"])},
                }
            )
            master_drift.append(
                {
                    "from_frame": 1,
                    "to_frame": i + 1,
                    "frame": i + 1,
                    "score": round(float(np.mean(list(metrics.values()))), 4),
                    "dimensions": dimensions,
                }
            )
        # Health is presented as stability (100 = no drift).
        values = [d["score"] for d in drift] or [0.0]
        mean_drift = float(np.mean(values))
        def stability(dimension: str) -> float:
            if not drift:
                return 100.0
            values = [d["dimensions"].get(dimension, {"score": 0.0})["score"] for d in drift]
            return round(100 * (1 - float(np.mean(values))), 1)

        health = {
            "Identity Stability": round(100 * (1 - mean_drift), 1) if drift else 100.0,
            "Motion Stability": stability("Character Scale Jump"),
            "Color Stability": stability("Color Drift"),
            "Lighting Stability": stability("Brightness Flicker"),
            "Background Stability": stability("Background Drift"),
        }
        health["Overall"] = round(float(np.mean(list(health.values()))), 1)
        frame_metrics = []
        for i, (record, feature) in enumerate(zip(records, feature_rows), 1):
            frame_metrics.append(
                {
                    "frame_id": record.frame_id,
                    "frame_number": i,
                    "brightness": round(feature["brightness"], 2),
                    "contrast": round(feature["contrast"], 2),
                    "color": [round(float(x), 2) for x in feature["color"]],
                }
            )
        bad_frames = sorted({frame for issue in issues for frame in (issue.get("start_frame"), issue.get("end_frame"))})
        return {
            "health": health,
            "drift": drift,
            "master_drift": master_drift,
            "issues": issues,
            "frames": frame_metrics,
            "bad_frames": bad_frames,
            "recommended_frames": bad_frames,
        }


def result_to_json(result: dict[str, Any]) -> str:
    """Serialize a result while handling numpy scalar values."""

    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        raise TypeError(type(value).__name__)

    return json.dumps(result, ensure_ascii=False, indent=2, default=default)
