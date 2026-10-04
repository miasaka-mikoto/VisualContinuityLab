"""Continuity analysis pipeline.

This module intentionally uses explainable image statistics rather than a
black-box model.  It is suitable for local QC, synthetic demos, and as a
foundation for future optional detectors (face landmarks/segmentation can be
added through ``FrameFeatures.extra`` without changing the report schema).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
import math

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from .image_metrics import (
    brightness,
    color_distribution,
    compare_feature_matches,
    contrast,
    edge_density,
    edge_map,
    histogram,
    histogram_distance,
    hamming_distance,
    image_size,
    optical_flow,
    perceptual_hash,
    sharpness,
    vector_distance,
)
from .models import (
    ContinuityReport,
    DriftDimension,
    DriftResult,
    FrameFeatures,
    FrameRecord,
    HealthScores,
    Issue,
    ROI,
    Sequence,
)
from .sequence import load_image, sequence_from_source


DIMENSIONS = (
    "face_region", "hair_silhouette", "hair_color", "costume_color",
    "body_proportion", "character_position", "character_scale",
    "camera_composition", "background", "lighting", "sharpness",
)

DIMENSION_LABELS = {
    "face_region": "Face region",
    "hair_silhouette": "Hair silhouette",
    "hair_color": "Hair color",
    "costume_color": "Costume color",
    "body_proportion": "Body proportion",
    "character_position": "Character position",
    "character_scale": "Character scale",
    "camera_composition": "Camera composition",
    "background": "Background",
    "lighting": "Lighting",
    "sharpness": "Sharpness",
}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return float(max(low, min(high, float(value))))
    except Exception:
        return low


def _mean_color(arr) -> tuple[float, float, float]:
    if np is None or arr is None or getattr(arr, "size", 0) == 0:
        return (0.0, 0.0, 0.0)
    value = np.mean(arr.reshape(-1, arr.shape[-1]), axis=0)
    return tuple(float(v) / 255.0 for v in value[:3])


def _resize_array(arr, width: int, height: int):
    if np is None:
        return arr
    if Image is not None:
        try:
            if arr.ndim == 2:
                out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
                return np.asarray(out, dtype=np.float32)
            out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
            return np.asarray(out, dtype=np.float32)
        except Exception:
            pass
    ys = np.linspace(0, max(0, arr.shape[0] - 1), height).astype(int)
    xs = np.linspace(0, max(0, arr.shape[1] - 1), width).astype(int)
    return arr[np.ix_(ys, xs)]


def _crop(arr, roi: ROI):
    if np is None:
        return arr
    h, w = arr.shape[:2]
    left, top, right, bottom = roi.as_pixels((w, h))
    return arr[top:bottom, left:right]


def _auto_roi(arr) -> ROI:
    """Infer a coarse foreground rectangle from border colour differences."""

    if np is None or arr is None or arr.ndim < 3 or arr.shape[0] < 4 or arr.shape[1] < 4:
        return ROI(0.2, 0.05, 0.6, 0.9, True, False, "auto-character")
    h, w = arr.shape[:2]
    border = np.concatenate((arr[: max(1, h // 20)].reshape(-1, 3),
                             arr[-max(1, h // 20):].reshape(-1, 3),
                             arr[:, : max(1, w // 20)].reshape(-1, 3),
                             arr[:, -max(1, w // 20):].reshape(-1, 3)), axis=0)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(arr.astype(np.float32) - bg, axis=2)
    # A low threshold intentionally includes soft silhouettes and anti-aliasing.
    threshold = max(12.0, float(np.percentile(dist, 72)) * 0.45)
    mask = dist > threshold
    # Remove tiny noise using a coarse downsample/upscale and bounding box.
    ys, xs = np.where(mask)
    if len(xs) < max(8, int(arr.shape[0] * arr.shape[1] * 0.002)):
        return ROI(0.2, 0.05, 0.6, 0.9, True, False, "auto-character")
    left, right = np.percentile(xs, [2, 98])
    top, bottom = np.percentile(ys, [2, 98])
    # Keep a small margin so face/hair edges are not clipped.
    margin_x, margin_y = max(2.0, (right - left) * 0.06), max(2.0, (bottom - top) * 0.04)
    left, right = max(0.0, left - margin_x), min(w - 1.0, right + margin_x)
    top, bottom = max(0.0, top - margin_y), min(h - 1.0, bottom + margin_y)
    roi = ROI(left / w, top / h, max(1.0, right - left) / w,
              max(1.0, bottom - top) / h, True, False, "auto-character")
    # Do not let an unusual background consume the whole frame.
    if roi.width > 0.95 or roi.height > 0.98:
        return ROI(0.2, 0.05, 0.6, 0.9, True, False, "auto-character")
    return roi.clamp()


def _flatten_gray(arr, width: int, height: int) -> list[float]:
    if np is None or arr is None or getattr(arr, "size", 0) == 0:
        return []
    if arr.ndim == 3:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    out = _resize_array(arr, width, height).astype(np.float32) / 255.0
    return [float(x) for x in out.reshape(-1)]


def _color_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if np is None:
        return 0.0
    return _clip(float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / math.sqrt(max(1, len(a)))))


def _foreground_stats(char, chosen_roi: ROI, image_shape: tuple[int, int]):
    """Estimate foreground centroid/area inside a locked ROI.

    This keeps position and scale diagnostics useful when the user intentionally
    locks one broad ROI for an entire shot: the ROI itself stays fixed, while
    the foreground silhouette can still move or grow within it.
    """
    if np is None or char is None or getattr(char, "size", 0) == 0:
        return ((chosen_roi.x + chosen_roi.width / 2.0, chosen_roi.y + chosen_roi.height / 2.0),
                chosen_roi.width * chosen_roi.height, chosen_roi.width / max(1e-6, chosen_roi.height))
    h, w = char.shape[:2]
    border = np.concatenate((char[: max(1, h // 12)].reshape(-1, 3),
                             char[-max(1, h // 12):].reshape(-1, 3),
                             char[:, : max(1, w // 12)].reshape(-1, 3),
                             char[:, -max(1, w // 12):].reshape(-1, 3)), axis=0)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(char.astype(np.float32) - bg, axis=2)
    threshold = max(10.0, float(np.percentile(dist, 65)) * 0.35)
    mask = dist > threshold
    ys, xs = np.where(mask)
    if len(xs) < max(6, int(w * h * 0.01)):
        return ((chosen_roi.x + chosen_roi.width / 2.0, chosen_roi.y + chosen_roi.height / 2.0),
                chosen_roi.width * chosen_roi.height, chosen_roi.width / max(1e-6, chosen_roi.height))
    left, right = float(xs.min()), float(xs.max() + 1)
    top, bottom = float(ys.min()), float(ys.max() + 1)
    # Avoid tiny isolated marks dominating scale/position.
    if (right - left) * (bottom - top) < 0.01 * w * h:
        return ((chosen_roi.x + chosen_roi.width / 2.0, chosen_roi.y + chosen_roi.height / 2.0),
                chosen_roi.width * chosen_roi.height, chosen_roi.width / max(1e-6, chosen_roi.height))
    cx, cy = float(xs.mean()), float(ys.mean())
    pos = (chosen_roi.x + (cx / max(1, w)) * chosen_roi.width,
           chosen_roi.y + (cy / max(1, h)) * chosen_roi.height)
    image_w, image_h = image_shape
    area = ((right - left) / max(1, w) * chosen_roi.width) * ((bottom - top) / max(1, h) * chosen_roi.height)
    aspect = ((right - left) / max(1e-6, bottom - top))
    return pos, _clip(area), aspect


def extract_features(image: Any, roi: ROI | None = None) -> FrameFeatures:
    """Extract all explainable features used by the analyzer."""

    loaded = load_image(image)
    if np is None:
        # The data model remains valid in a no-NumPy environment, although
        # feature quality is necessarily limited.
        size = image_size(loaded)
        return FrameFeatures(image_size=size, brightness=brightness(loaded), contrast=contrast(loaded),
                             histogram=histogram(loaded), color_distribution=color_distribution(loaded),
                             perceptual_hash=perceptual_hash(loaded), roi=roi)
    arr = np.asarray(loaded)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    h, w = arr.shape[:2]
    chosen_roi = (roi.clamp() if roi else _auto_roi(arr))
    char = _crop(arr, chosen_roi)
    # Regions are relative to the inferred/locked character crop.
    ch, cw = char.shape[:2]
    upper = char[: max(1, int(ch * 0.32))]
    hair = char[: max(1, int(ch * 0.28))]
    costume = char[int(ch * 0.42):] if ch > 2 else char
    # Outside ROI is a robust background proxy.  Resize full frame signatures to
    # fixed dimensions so changing source resolutions does not cause false drift.
    background = arr.copy()
    left, top, right, bottom = chosen_roi.as_pixels((w, h))
    background[top:bottom, left:right] = np.median(arr, axis=(0, 1)).astype(arr.dtype)
    edges = edge_map(arr)
    edge_sig = _flatten_gray(_resize_array(edges, 16, 16), 16, 16)
    # Hair silhouette captures edges and luminance, not only colour.
    hair_edges = edge_map(hair)
    hair_sig = _flatten_gray(_resize_array(hair_edges, 16, 8), 16, 8)
    face_sig = _flatten_gray(upper, 16, 16)
    pos, scale, aspect = _foreground_stats(char, chosen_roi, (w, h))
    return FrameFeatures(
        image_size=(int(w), int(h)),
        brightness=brightness(arr),
        contrast=contrast(arr),
        histogram=histogram(arr),
        color_distribution=color_distribution(arr),
        perceptual_hash=perceptual_hash(arr),
        edge_density=edge_density(arr),
        edge_signature=edge_sig,
        sharpness=sharpness(arr),
        character_position=(float(pos[0]), float(pos[1])),
        character_scale=float(scale),
        body_aspect=float(aspect),
        face_signature=face_sig,
        hair_signature=hair_sig,
        hair_color=_mean_color(hair),
        costume_color=_mean_color(costume),
        background_histogram=histogram(background),
        roi=chosen_roi,
        extra={"roi_source": "locked" if roi and roi.locked else ("provided" if roi else "auto")},
    )


def _severity(score: float, low: float, medium: float) -> str:
    score = _clip(score)
    if score < low:
        return "low"
    if score < medium:
        return "medium"
    return "high"


def _dimension(score: float, name: str, *, delta: float | None = None,
               evidence: Mapping[str, Any] | None = None,
               low: float = 0.15, medium: float = 0.35) -> DriftDimension:
    return DriftDimension(name=name, score=_clip(score), severity=_severity(score, low, medium),
                          delta=delta, evidence=dict(evidence or {}))


def compare_features(
    a: FrameFeatures,
    b: FrameFeatures,
    *,
    image_a: Any | None = None,
    image_b: Any | None = None,
    low_threshold: float = 0.15,
    medium_threshold: float = 0.35,
    comparison: str = "adjacent",
    include_optical_flow: bool = True,
) -> DriftResult:
    """Compare two feature sets and retain evidence for every dimension."""

    def ratio_delta(x: float, y: float) -> float:
        if x <= 1e-8 or y <= 1e-8:
            return _clip(abs(x - y))
        return _clip(abs(math.log(x / y)) / math.log(2.0))

    dims: dict[str, DriftDimension] = {
        "face_region": _dimension(vector_distance(a.face_signature, b.face_signature), "face_region",
                                   evidence={"method": "normalized_face_patch", "feature_length": len(a.face_signature)}),
        "hair_silhouette": _dimension(vector_distance(a.hair_signature, b.hair_signature), "hair_silhouette",
                                       evidence={"method": "edge_patch", "feature_length": len(a.hair_signature)}),
        "hair_color": _dimension(_color_distance(a.hair_color, b.hair_color), "hair_color",
                                  evidence={"from_rgb": a.hair_color, "to_rgb": b.hair_color}),
        "costume_color": _dimension(_color_distance(a.costume_color, b.costume_color), "costume_color",
                                     evidence={"from_rgb": a.costume_color, "to_rgb": b.costume_color}),
        "body_proportion": _dimension(ratio_delta(a.body_aspect, b.body_aspect), "body_proportion",
                                       delta=b.body_aspect - a.body_aspect,
                                       evidence={"from_aspect": a.body_aspect, "to_aspect": b.body_aspect}),
        "character_position": _dimension(vector_distance(a.character_position, b.character_position) * 2.0,
                                           "character_position",
                                           evidence={"from_center": a.character_position, "to_center": b.character_position}),
        "character_scale": _dimension(ratio_delta(a.character_scale, b.character_scale), "character_scale",
                                       delta=b.character_scale - a.character_scale,
                                       evidence={"from_area": a.character_scale, "to_area": b.character_scale}),
        "camera_composition": _dimension(vector_distance(a.edge_signature, b.edge_signature), "camera_composition",
                                           evidence={"from_edge_density": a.edge_density, "to_edge_density": b.edge_density}),
        "background": _dimension(histogram_distance(a.background_histogram, b.background_histogram), "background",
                                   evidence={"method": "outside_roi_histogram"}),
        # Exposure changes are intentionally amplified: a modest normalized
        # luminance jump can still be a very visible animation flicker.  The
        # multiplier is calibrated against the synthetic demo (1.30x/0.68x)
        # while remaining bounded to 0..1 for real footage.
        "lighting": _dimension(_clip(abs(a.brightness - b.brightness) * 3.5 + abs(a.contrast - b.contrast) * 1.5), "lighting",
                                delta=b.brightness - a.brightness,
                                evidence={"from_brightness": a.brightness, "to_brightness": b.brightness,
                                          "from_contrast": a.contrast, "to_contrast": b.contrast}),
        "sharpness": _dimension(_clip(abs(a.sharpness - b.sharpness) * 2.0), "sharpness",
                                 delta=b.sharpness - a.sharpness,
                                 evidence={"from_sharpness": a.sharpness, "to_sharpness": b.sharpness}),
    }
    # Apply caller thresholds (the helper's defaults are intentionally not
    # mutable after creation).
    for name, dim in list(dims.items()):
        dims[name] = DriftDimension(dim.name, dim.score, _severity(dim.score, low_threshold, medium_threshold), dim.delta, dim.evidence)
    scores = [d.score for d in dims.values()]
    overall = float(sum(scores) / max(1, len(scores)))
    overall_severity = _severity(max(overall, max(scores, default=0.0) * 0.75), low_threshold, medium_threshold)
    evidence: dict[str, Any] = {
        "dimensions": {name: dim.evidence for name, dim in dims.items()},
        "perceptual_hash_distance": hamming_distance(a.perceptual_hash, b.perceptual_hash),
        "image_size_from": a.image_size,
        "image_size_to": b.image_size,
    }
    if image_a is not None and image_b is not None:
        try:
            evidence["feature_matching"] = compare_feature_matches(image_a, image_b)
            if include_optical_flow:
                evidence["optical_flow"] = optical_flow(image_a, image_b)
        except Exception as exc:  # Metrics should not make the whole report fail.
            evidence["metric_warning"] = str(exc)
    return DriftResult(from_frame="", to_frame="", dimensions=dims, overall_score=overall,
                       severity=overall_severity, evidence=evidence, comparison=comparison)


@dataclass
class AnalyzerConfig:
    low_threshold: float = 0.15
    medium_threshold: float = 0.35
    include_optical_flow: bool = True
    include_features_in_report: bool = True


class ContinuityAnalyzer:
    """Analyze one sequence and produce an explainable :class:`ContinuityReport`."""

    def __init__(self, config: AnalyzerConfig | None = None, **kwargs: Any):
        if config is None:
            config = AnalyzerConfig(**{k: v for k, v in kwargs.items() if k in AnalyzerConfig.__annotations__})
        self.config = config

    # Name used by the desktop adapter and early plugin prototypes.
    def analyze_sequence(self, source, **kwargs: Any) -> ContinuityReport:
        return self.analyze(source, **kwargs)

    def prepare_sequence(self, source: Sequence | str | Path | Iterable[str | Path], **kwargs: Any) -> Sequence:
        if isinstance(source, Sequence):
            # ``roi`` is accepted by the UI adapter as a convenient tuple in
            # addition to the canonical ROI model.
            roi = kwargs.get("roi")
            if roi is not None:
                source.set_roi(roi if isinstance(roi, ROI) else ROI(*roi, normalized=True), lock=True)
            return source
        # Accept an already-loaded PIL/NumPy image as a one-frame sequence.
        # Keeping the object in metadata avoids writing a surprising temporary
        # file; it is used only for the duration of ``analyze``.
        if (Image is not None and isinstance(source, Image.Image)) or (np is not None and isinstance(source, np.ndarray)):
            return Sequence(sequence_id=kwargs.get("sequence_id", "single-image"), frames=[
                FrameRecord(frame_id="frame_0001", path=None, frame_number=1,
                            metadata={"image": source})
            ])
        if isinstance(source, (str, Path)):
            roi = kwargs.pop("roi", None)
            result = sequence_from_source(source, **kwargs)
            if roi is not None:
                result.set_roi(roi if isinstance(roi, ROI) else ROI(*roi, normalized=True), lock=True)
            return result
        # Integrations often pass a list of FrameRecord objects or dictionaries
        # rather than file paths.  Preserve their IDs/timestamps instead of
        # forcing callers through a temporary manifest.
        values = list(source)
        if values and all(isinstance(v, FrameRecord) or isinstance(v, Mapping) for v in values):
            frames = [v if isinstance(v, FrameRecord) else FrameRecord.from_dict(v) for v in values]
            result = Sequence(sequence_id=kwargs.get("sequence_id", "sequence"), frames=frames,
                              shot=kwargs.get("shot"), character=kwargs.get("character"))
            roi = kwargs.get("roi")
            if roi is not None:
                result.set_roi(roi if isinstance(roi, ROI) else ROI(*roi, normalized=True), lock=True)
            return result
        return self._from_iterable(values, **kwargs)

    @staticmethod
    def _from_iterable(paths: Iterable[str | Path], **kwargs: Any) -> Sequence:
        from .sequence import sequence_from_paths
        return sequence_from_paths(paths, **kwargs)

    def _load_features(self, sequence: Sequence) -> tuple[dict[str, FrameFeatures], dict[str, Any]]:
        features: dict[str, FrameFeatures] = {}
        images: dict[str, Any] = {}
        for frame in sequence.frames:
            image_source = frame.metadata.get("image") if isinstance(frame.metadata, dict) else None
            if not frame.path and image_source is None:
                frame.status = "missing-image"
                continue
            try:
                image = load_image(frame.path if frame.path else image_source)
                images[frame.frame_id] = image
                features[frame.frame_id] = extract_features(image, frame.roi)
                frame.status = "analysed"
                # Preserve auto ROI for UI preview while respecting a locked ROI.
                if frame.roi is None or not frame.roi.locked:
                    features_roi = features[frame.frame_id].roi
                    frame.metadata["detected_roi"] = features_roi.to_dict() if features_roi else None
            except Exception as exc:
                frame.status = "error"
                frame.metadata["analysis_error"] = str(exc)
        return features, images

    def _compare(self, first: FrameRecord, second: FrameRecord,
                 features: Mapping[str, FrameFeatures], images: Mapping[str, Any], comparison: str) -> DriftResult:
        result = compare_features(features[first.frame_id], features[second.frame_id],
                                   image_a=images.get(first.frame_id), image_b=images.get(second.frame_id),
                                   low_threshold=self.config.low_threshold,
                                   medium_threshold=self.config.medium_threshold,
                                   comparison=comparison,
                                   include_optical_flow=self.config.include_optical_flow)
        result.from_frame, result.to_frame = first.frame_id, second.frame_id
        return result

    def analyze(self, source: Sequence | str | Path | Iterable[str | Path], *,
                master_reference: str | int | None = None, master_index: int | None = None,
                master: str | int | None = None, **source_kwargs: Any) -> ContinuityReport:
        # Alias names are kept for the desktop adapter and early FrameForge
        # prototypes.  Explicit ``master_reference`` wins when both are set.
        if master_reference is None:
            master_reference = master_index if master_index is not None else master
        sequence = self.prepare_sequence(source, **source_kwargs)
        if not sequence.frames:
            sequence.status = "empty"
            return ContinuityReport(sequence_id=sequence.sequence_id,
                                    metadata={"warning": "Sequence contains no frames"})
        features, images = self._load_features(sequence)
        valid = [f for f in sequence.frames if f.frame_id in features]
        adjacent: list[DriftResult] = []
        for first, second in zip(valid, valid[1:]):
            adjacent.append(self._compare(first, second, features, images, "adjacent"))
        master_id: str | None
        if master_reference is None:
            master_id = sequence.master_reference or (valid[0].frame_id if valid else None)
        elif isinstance(master_reference, int):
            master_id = valid[master_reference].frame_id if 0 <= master_reference < len(valid) else None
        else:
            master_id = str(master_reference)
        master_results: list[DriftResult] = []
        # Manifests sometimes store a master image path while feature maps use
        # frame IDs.  Resolve by path/stem as a compatibility convenience.
        if master_id not in features and master_id is not None:
            master_name = Path(str(master_id)).name
            match = next((f for f in valid if Path(str(f.path or "")).name == master_name or Path(str(f.frame_id)).stem == Path(master_name).stem), None)
            if match:
                master_id = match.frame_id
        if master_id in features:
            sequence.master_reference = master_id
            master = next(f for f in valid if f.frame_id == master_id)
            for target in valid:
                if target.frame_id != master_id:
                    master_results.append(self._compare(master, target, features, images, "master"))
        timeline = self._timeline(adjacent, valid)
        issues = self._issues(adjacent, valid)
        health = self._health(adjacent)
        bad = self._bad_frames(issues, sequence)
        recommended = self._recommended_frames(bad, valid)
        sequence.status = "analysed"
        metadata = {
            "analyzer": "Visual Continuity Lab statistical analyzer",
            "config": {"low_threshold": self.config.low_threshold, "medium_threshold": self.config.medium_threshold},
            "master_reference": master_id,
            "frame_count": len(sequence.frames),
            "analysed_frame_count": len(valid),
        }
        return ContinuityReport(sequence_id=sequence.sequence_id,
                                adjacent_results=adjacent, master_results=master_results,
                                issues=issues, timeline=timeline, health=health,
                                bad_frames=bad, recommended_frames=recommended,
                                manual_labels=list(sequence.manual_labels),
                                frame_features=features if self.config.include_features_in_report else {},
                                metadata=metadata)

    def _timeline(self, results: list[DriftResult], frames: list[FrameRecord]) -> dict[str, list[dict[str, Any]]]:
        timeline: dict[str, list[dict[str, Any]]] = {name: [] for name in (*DIMENSIONS, "overall")}
        number_by_id = {f.frame_id: f.frame_number for f in frames}
        if frames:
            # Frame one is a natural zero baseline for plotting.
            for name in timeline:
                timeline[name].append({"frame_id": frames[0].frame_id, "frame_number": number_by_id.get(frames[0].frame_id, 0), "score": 0.0, "severity": "low"})
        for result in results:
            frame_id, number = result.to_frame, number_by_id.get(result.to_frame, 0)
            for name in DIMENSIONS:
                dim = result.dimensions.get(name)
                if dim:
                    timeline[name].append({"frame_id": frame_id, "frame_number": number, "score": dim.score, "severity": dim.severity})
            timeline["overall"].append({"frame_id": frame_id, "frame_number": number, "score": result.overall_score, "severity": result.severity})
        return timeline

    def _issues(self, results: list[DriftResult], frames: list[FrameRecord]) -> list[Issue]:
        if not results:
            return []
        frame_index = {f.frame_id: i for i, f in enumerate(frames)}
        out: list[Issue] = []
        issue_counter = 1
        for name in DIMENSIONS:
            candidates = [
                (i, result) for i, result in enumerate(results)
                if result.dimensions.get(name) and result.dimensions[name].score >= self.config.low_threshold
            ]
            if not candidates:
                continue
            groups: list[list[tuple[int, DriftResult]]] = []
            for index, result in candidates:
                if not groups or index > groups[-1][-1][0] + 1:
                    groups.append([])
                groups[-1].append((index, result))
            for group in groups:
                first_result, last_result = group[0][1], group[-1][1]
                dimensions = [r.dimensions[name] for _, r in group]
                worst = max(dimensions, key=lambda d: d.score)
                severity = "high" if any(d.severity == "high" for d in dimensions) else "medium"
                evidence = {
                    "dimension": name,
                    "label": DIMENSION_LABELS.get(name, name),
                    "max_score": worst.score,
                    "score": worst.score,
                    "from_frame": first_result.from_frame,
                    "to_frame": last_result.to_frame,
                    "threshold": self.config.low_threshold,
                    "samples": [
                        {"from_frame": r.from_frame, "to_frame": r.to_frame, "score": d.score,
                         "severity": d.severity, "evidence": d.evidence}
                        for (_, r), d in zip(group, dimensions)
                    ],
                }
                out.append(Issue(
                    issue_id=f"issue-{issue_counter:04d}",
                    type=name,
                    start_frame=first_result.from_frame,
                    end_frame=last_result.to_frame,
                    severity=severity,
                    evidence=evidence,
                    notes=f"{DIMENSION_LABELS.get(name, name)} changed at {first_result.to_frame}.",
                ))
                issue_counter += 1
        return out

    def _health(self, results: list[DriftResult]) -> HealthScores:
        if not results:
            return HealthScores(sample_count=0)
        def avg(names: Iterable[str]) -> float:
            vals = [r.dimensions[n].score for r in results for n in names if n in r.dimensions]
            return sum(vals) / max(1, len(vals))
        identity = avg(("face_region", "hair_silhouette", "hair_color", "costume_color", "body_proportion"))
        motion = avg(("character_position", "character_scale", "camera_composition"))
        color = avg(("hair_color", "costume_color"))
        lighting = avg(("lighting", "sharpness"))
        background = avg(("background", "camera_composition"))
        # Weighted equally: the report exposes dimensions instead of hiding
        # everything behind one score.
        overall_drift = (identity + motion + color + lighting + background) / 5.0
        return HealthScores(
            identity_stability=round(_clip(100.0 * (1.0 - identity), 0.0, 100.0), 2),
            motion_stability=round(_clip(100.0 * (1.0 - motion), 0.0, 100.0), 2),
            color_stability=round(_clip(100.0 * (1.0 - color), 0.0, 100.0), 2),
            lighting_stability=round(_clip(100.0 * (1.0 - lighting), 0.0, 100.0), 2),
            background_stability=round(_clip(100.0 * (1.0 - background), 0.0, 100.0), 2),
            overall=round(_clip(100.0 * (1.0 - overall_drift), 0.0, 100.0), 2),
            sample_count=len(results),
        )

    @staticmethod
    def _bad_frames(issues: list[Issue], sequence: Sequence) -> list[str]:
        bad = []
        for issue in issues:
            if issue.severity in {"medium", "high"}:
                bad.extend((issue.start_frame, issue.end_frame))
        for label in sequence.manual_labels:
            if label.get("label") and label.get("label") != "Good":
                bad.append(str(label.get("frame_id")))
        # Preserve sequence order and remove duplicates.
        order = {f.frame_id: i for i, f in enumerate(sequence.frames)}
        return sorted({b for b in bad if b}, key=lambda x: order.get(x, 10**9))

    @staticmethod
    def _recommended_frames(bad: list[str], frames: list[FrameRecord]) -> list[str]:
        if not bad:
            return []
        order = {f.frame_id: i for i, f in enumerate(frames)}
        rec = set(bad)
        for frame_id in bad:
            i = order.get(frame_id)
            if i is not None:
                if i > 0:
                    rec.add(frames[i - 1].frame_id)
                if i + 1 < len(frames):
                    rec.add(frames[i + 1].frame_id)
        return sorted(rec, key=lambda x: order.get(x, 10**9))


def analyze_sequence(source: Sequence | str | Path | Iterable[str | Path], **kwargs: Any) -> ContinuityReport:
    """Functional convenience wrapper around :class:`ContinuityAnalyzer`."""

    config_keys = {"low_threshold", "medium_threshold", "include_optical_flow", "include_features_in_report"}
    config = {k: kwargs.pop(k) for k in tuple(kwargs) if k in config_keys}
    analyzer = ContinuityAnalyzer(**config)
    return analyzer.analyze(source, **kwargs)


__all__ = [
    "Analyzer", "AnalyzerConfig", "ContinuityAnalyzer", "DIMENSIONS", "DIMENSION_LABELS",
    "analyze_sequence", "compare_features", "extract_features",
]

# Backwards-compatible short name for integrations that discovered an
# ``Analyzer`` class before the product name was finalized.
Analyzer = ContinuityAnalyzer
