"""Dependency-light report engine used by the deterministic demo CLI.

The desktop core owns the richer public API.  This module intentionally keeps a
small, stable adapter for CI and packaged-demo checks: it reads the mock
manifest/ROIs, computes explainable image differences, and emits JSON-friendly
records with evidence.  No model or network call is involved.
"""

from __future__ import annotations

import html
import json
import math
import os
import csv
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:  # Plotting is optional; Pillow fallback is below.
    # Managed workspaces often make /root read-only; point Matplotlib's cache
    # at a writable temp location before importing it to avoid noisy warnings.
    _mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "vclab-matplotlib"
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _box(value: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return (0, 0, width, height)
    x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(1, min(width, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(1, min(height, y2))))
    return (x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))


def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def _resize(image: np.ndarray, size: tuple[int, int] = (48, 48)) -> np.ndarray:
    if image.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.float32)
    return np.asarray(Image.fromarray(np.uint8(np.clip(image, 0, 255))).resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(_resize(a) - _resize(b))) / 255.0)


def _structural_mae(a: np.ndarray, b: np.ndarray) -> float:
    """Patch difference after luminance normalization (shape, not exposure)."""
    aa, bb = _resize(a), _resize(b)
    aa = aa[..., 0] * .299 + aa[..., 1] * .587 + aa[..., 2] * .114
    bb = bb[..., 0] * .299 + bb[..., 1] * .587 + bb[..., 2] * .114
    aa = (aa - aa.mean()) / max(1.0, float(aa.std()))
    bb = (bb - bb.mean()) / max(1.0, float(bb.std()))
    return float(min(1.0, np.mean(np.abs(aa - bb)) / 4.0))


def _mean_rgb(image: np.ndarray) -> np.ndarray:
    return image.reshape(-1, 3).mean(axis=0) if image.size else np.zeros(3)


def _brightness(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    return float(np.mean(image[..., 0] * .299 + image[..., 1] * .587 + image[..., 2] * .114) / 255.0)


def _hash(image: np.ndarray) -> str:
    gray = np.asarray(Image.fromarray(np.uint8(np.clip(image, 0, 255))).convert("L").resize((8, 8)), dtype=float)
    return "".join("1" if value >= gray.mean() else "0" for value in gray.reshape(-1))


def _histogram(image: np.ndarray, bins: int = 16) -> list[float]:
    values: list[float] = []
    for channel in range(3):
        hist, _ = np.histogram(image[..., channel], bins=bins, range=(0, 256))
        values.extend((hist / max(1, hist.sum())).round(6).tolist())
    return values


def _edge_density(image: np.ndarray) -> float:
    gray = image[..., 0] * .299 + image[..., 1] * .587 + image[..., 2] * .114
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    return float((np.mean(np.abs(np.diff(gray, axis=0)) > 18) + np.mean(np.abs(np.diff(gray, axis=1)) > 18)) / 2.0)


def _hamming(a: str, b: str) -> float:
    return sum(x != y for x, y in zip(a, b)) / max(1, len(a))


def _chromatic_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compare colour ratios so exposure changes do not masquerade as hue drift."""
    aa = np.asarray(a, dtype=float); bb = np.asarray(b, dtype=float)
    aa = aa / max(1e-9, float(aa.sum())); bb = bb / max(1e-9, float(bb.sum()))
    return float(np.linalg.norm(aa - bb) / math.sqrt(3.0))


def _severity(value: float) -> str:
    if value >= .18:
        return "High"
    if value >= .08:
        return "Medium"
    return "Low"


def _load(source: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(source)
    # Keep the standalone CLI useful for real video input too.  The desktop
    # core has a richer extractor; this small adapter only needs sequential
    # RGB PNGs and deliberately leaves the source video untouched.
    if source.is_file() and source.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - minimal installs
            raise RuntimeError("Video analysis requires OpenCV (cv2)") from exc
        target = Path(tempfile.mkdtemp(prefix="vcl_demo_video_"))
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {source}")
        index = 0
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                # Convert once so the rest of this module can keep its RGB
                # contract (OpenCV decodes frames as BGR).
                Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(target / f"frame_{index + 1:06d}.png")
                index += 1
        finally:
            capture.release()
        if index == 0:
            raise ValueError(f"Video has no decodable frames: {source}")
        source = target
    explicit_manifest = source.is_file() and source.suffix.lower() == ".json"
    single_image = source.is_file() and source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
    root = source.parent if source.is_file() else source
    manifest_path = source if explicit_manifest else root / "sequence.json"
    # A direct image path is intentionally isolated from a neighbouring
    # sequence manifest; callers asking for one image should not silently load
    # every sibling frame.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() and not single_image else {}
    entries = ([{"frame_id": source.stem, "frame_number": 1, "file": source.name}] if single_image else list(manifest.get("frames", [])))
    if not entries:
        def natural_key(path: Path) -> list[Any]:
            return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]
        paths = sorted((p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif"}), key=natural_key)
        entries = [{"frame_id": f"f{i:04d}", "frame_number": i, "file": str(p.relative_to(root))} for i, p in enumerate(paths, 1)]
    frames: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, 1):
        path = root / str(entry.get("file", ""))
        if not path.exists():
            path = root / "frames" / Path(str(entry.get("file", ""))).name
        if not path.exists():
            continue
        with Image.open(path) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        h, w = image.shape[:2]
        roi = dict(entry.get("roi") or {})
        roi.setdefault("character", [int(w * .3), int(h * .15), int(w * .7), int(h * .86)])
        for key in ("face", "hair", "costume"):
            roi.setdefault(key, roi["character"])
        frames.append({"number": int(entry.get("frame_number", index)), "frame_id": str(entry.get("frame_id", f"f{index:04d}")), "path": str(path), "image": image, "roi": roi, "entry": entry})
    if not frames:
        raise FileNotFoundError(f"No image frames found under {source}")
    frames.sort(key=lambda frame: frame["number"])
    manifest.setdefault("sequence_id", root.name)
    manifest.setdefault("name", root.name)
    manifest.setdefault("fps", 0.0)
    return manifest, frames


def _frame_metric(frame: dict[str, Any]) -> dict[str, Any]:
    image = frame["image"]
    h, w = image.shape[:2]
    boxes = {key: _box(frame["roi"].get(key), w, h) for key in ("character", "face", "hair", "costume")}
    character = _crop(image, boxes["character"])
    face = _crop(image, boxes["face"])
    hair = _crop(image, boxes["hair"])
    costume = _crop(image, boxes["costume"])
    x1, y1, x2, y2 = boxes["character"]
    rel = {"x": x1 / w, "y": y1 / h, "width": (x2 - x1) / w, "height": (y2 - y1) / h, "area": (x2 - x1) * (y2 - y1) / (w * h), "center_x": (x1 + x2) / (2 * w), "center_y": (y1 + y2) / (2 * h)}
    outside = np.concatenate((image[: max(1, int(h * .22))], image[int(h * .87) :]), axis=0)
    return {
        "frame_number": frame["number"], "frame_id": frame["frame_id"],
        "timestamp": frame["entry"].get("timestamp"), "shot": frame["entry"].get("shot"),
        "character_name": frame["entry"].get("character"), "character_id": frame["entry"].get("character"),
        "status": frame["entry"].get("status", "loaded"),
        "image_size": {"width": w, "height": h},
        "brightness": round(_brightness(image), 6),
        # Lighting evidence is measured inside the locked character ROI so a
        # deliberate background palette change is not mislabeled as a light
        # flicker.
        "character_brightness": round(_brightness(character), 6),
        "contrast": round(float(np.std(image[..., 0] * .299 + image[..., 1] * .587 + image[..., 2] * .114) / 255.0), 6),
        "color_distribution": [round(float(v) / 255.0, 6) for v in _mean_rgb(image)],
        "histogram": _histogram(image),
        "perceptual_hash": _hash(image),
        "edge_density": round(_edge_density(image), 6),
        "character": {
            "bbox": list(boxes["character"]), "relative_bbox": {k: round(float(v), 6) for k, v in rel.items()},
            "center": [round(rel["center_x"], 6), round(rel["center_y"], 6)],
            "scale": round(math.sqrt(rel["area"] / .12), 6),
            "face_color": [round(float(v) / 255.0, 6) for v in _mean_rgb(face)],
            "hair_color": [round(float(v) / 255.0, 6) for v in _mean_rgb(hair)],
            "costume_color": [round(float(v) / 255.0, 6) for v in _mean_rgb(costume)],
            "face_hash": _hash(face), "hair_hash": _hash(hair), "costume_hash": _hash(costume),
        },
        "_face": _resize(face), "_hair": _resize(hair), "_costume": _resize(costume), "_outside": _resize(outside, (64, 32)),
    }


def _drift(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    pc, cc = previous["character"], current["character"]
    color = _chromatic_distance(cc["costume_color"], pc["costume_color"])
    hair_color = _chromatic_distance(cc["hair_color"], pc["hair_color"])
    scale = abs(cc["relative_bbox"]["area"] - pc["relative_bbox"]["area"]) / max(1e-9, pc["relative_bbox"]["area"])
    position = math.hypot(cc["center"][0] - pc["center"][0], cc["center"][1] - pc["center"][1]) * 2.5
    values = {
        "face_drift": _structural_mae(current["_face"], previous["_face"]),
        "hair_drift": max(_structural_mae(current["_hair"], previous["_hair"]), _hamming(cc["hair_hash"], pc["hair_hash"])),
        "hair_color_drift": hair_color,
        "color_drift": max(color, _chromatic_distance(current["color_distribution"], previous["color_distribution"])),
        "body_proportion_drift": min(1.0, abs(cc["relative_bbox"]["width"] - pc["relative_bbox"]["width"]) + abs(cc["relative_bbox"]["height"] - pc["relative_bbox"]["height"])),
        "character_position_drift": min(1.0, position),
        "character_scale_jump": min(1.0, scale),
        "background_drift": _mae(current["_outside"], previous["_outside"]),
        "brightness_flicker": min(1.0, abs(current["character_brightness"] - previous["character_brightness"]) * 3.0),
        "sharpness_drift": 0.0,
    }
    evidence = {
        "costume_color_distance": round(color, 6), "hair_color_distance": round(hair_color, 6),
        "scale_ratio_delta": round(scale, 6), "position_delta": round(position, 6),
        "background_mae": round(values["background_drift"], 6), "brightness_delta": round(abs(current["character_brightness"] - previous["character_brightness"]), 6),
    }
    return {"from_frame": previous["frame_number"], "to_frame": current["frame_number"], "frame_id": current["frame_id"], "metrics": {key: {"value": round(float(value), 6), "severity": _severity(float(value))} for key, value in values.items()}, "evidence": evidence}


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_strip(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def analyze_sequence(source: str | Path) -> dict[str, Any]:
    manifest, frames = _load(source)
    metrics = [_frame_metric(frame) for frame in frames]
    drifts = [_drift(metrics[index - 1], metrics[index]) for index in range(1, len(metrics))]
    master = frames[0]
    master_reference = str(manifest.get("master_reference", ""))
    if master_reference:
        reference_name = Path(master_reference).name
        for candidate in frames:
            if Path(candidate["path"]).name == reference_name or candidate["entry"].get("file") == master_reference:
                master = candidate
                break
    issue_labels = {
        "face_drift": "Face region changed", "hair_drift": "Hair silhouette changed", "hair_color_drift": "Hair colour changed", "color_drift": "Costume colour drift", "body_proportion_drift": "Body proportion changed", "character_position_drift": "Character position jump", "character_scale_jump": "Character scale jump", "background_drift": "Background changed", "brightness_flicker": "Brightness flicker", "sharpness_drift": "Sharpness changed",
    }
    issues = []
    for drift in drifts:
        for name, item in drift["metrics"].items():
            if item["severity"] == "Low":
                continue
            issues.append({"issue_id": f"issue-{drift['from_frame']:04d}-{drift['to_frame']:04d}-{name}", "type": name, "start_frame": drift["from_frame"], "end_frame": drift["to_frame"], "severity": item["severity"], "evidence": {"metric": item["value"], **drift["evidence"]}, "notes": issue_labels.get(name, name.replace("_", " "))})
    def stability(names: tuple[str, ...]) -> float:
        values = [drift["metrics"][name]["value"] for drift in drifts for name in names]
        return round(max(0.0, min(100.0, 100.0 * (1.0 - (float(np.mean(values)) if values else 0.0)))), 2)
    health = {"identity_stability": stability(("face_drift", "hair_drift", "hair_color_drift", "color_drift", "body_proportion_drift")), "motion_stability": stability(("character_position_drift", "character_scale_jump")), "color_stability": stability(("color_drift", "hair_color_drift")), "lighting_stability": stability(("brightness_flicker", "sharpness_drift")), "background_stability": stability(("background_drift",)),}
    health["overall"] = round(float(np.mean(list(health.values()))), 2)
    bad: dict[int, str] = {}
    ranks = {"Low": 1, "Medium": 2, "High": 3}
    for issue in issues:
        for number in (issue["start_frame"], issue["end_frame"]):
            if ranks[issue["severity"]] > ranks.get(bad.get(number, "Low"), 0):
                bad[number] = issue["severity"]
    events = manifest.get("demo_markers", {})
    if not events:
        events = {key: item.get("frames", []) for key, item in manifest.get("intentional_events", {}).items() if isinstance(item, dict)}
    timeline = [{"from_frame": drift["from_frame"], "to_frame": drift["to_frame"], **{key: item["value"] for key, item in drift["metrics"].items()}} for drift in drifts]
    report = _strip({"schema_version": "1.0", "report_type": "Shot Continuity Report", "sequence": {"sequence_id": manifest.get("sequence_id", Path(source).stem), "name": manifest.get("name", Path(source).stem), "frame_count": len(frames), "width": int(frames[0]["image"].shape[1]), "height": int(frames[0]["image"].shape[0]), "fps": manifest.get("fps", 0), "master_reference": manifest.get("master_reference")}, "frames": metrics, "adjacent_consistency": drifts, "master_consistency": {"master_frame": master["number"], "per_frame": [{"frame": frame["number"], "image_mae": _mae(frame["image"], master["image"])} for frame in frames]}, "drift_timeline": timeline, "issues": issues, "bad_frames": [{"frame": key, "severity": value} for key, value in sorted(bad.items())], "recommended_frames_to_regenerate": [key for key, value in sorted(bad.items()) if value in ("Medium", "High")], "sequence_health": health, "manual_labels": [], "demo_markers": events})
    # Compatibility aliases let the standalone report be consumed by the
    # desktop adapter and older FrameForge-oriented scripts without changing
    # the clearer names used by this demo.
    report["sequence_id"] = report["sequence"]["sequence_id"]
    report["adjacent_results"] = report["adjacent_consistency"]
    report["timeline"] = report["drift_timeline"]
    report["health"] = report["sequence_health"]
    report["recommended_frames"] = report["recommended_frames_to_regenerate"]
    return report


def _plot(report: dict[str, Any], path: Path) -> None:
    timeline = report["drift_timeline"]
    x = [item["to_frame"] for item in timeline]
    series = {"color": [item["color_drift"] for item in timeline], "scale": [item["character_scale_jump"] for item in timeline], "brightness": [item["brightness_flicker"] for item in timeline], "background": [item["background_drift"] for item in timeline]}
    if plt is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for label, values in series.items(): axes[0].plot(x, values, marker="o", label=label)
        axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("drift (0–1)"); axes[0].grid(alpha=.25); axes[0].legend(ncol=4, fontsize=8)
        axes[1].bar(list(report["sequence_health"]), list(report["sequence_health"].values()), color="#4575b4"); axes[1].set_ylim(0, 100); axes[1].tick_params(axis="x", rotation=20, labelsize=7)
        fig.suptitle("Visual Continuity Lab — Shot Continuity Report"); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); return
    image = Image.new("RGB", (1100, 620), "white"); draw = ImageDraw.Draw(image); draw.text((25, 20), "Visual Continuity Lab — Shot Continuity Report", fill="#18243a")
    left, top, right, bottom = 60, 70, 1060, 380; draw.rectangle((left, top, right, bottom), outline="#8190a5", width=2)
    colors = {"color": "#d95f59", "scale": "#4c78a8", "brightness": "#f2a541", "background": "#59a14f"}
    for label, values in series.items():
        points = [(left + (right - left) * i / max(1, len(values) - 1), bottom - (bottom - top) * value) for i, value in enumerate(values)]
        if len(points) > 1: draw.line(points, fill=colors[label], width=3)
    image.save(path)


def write_report(report: dict[str, Any], output: str | Path) -> dict[str, str]:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    json_path, png_path, html_path, md_path = output / "analysis.json", output / "drift_report.png", output / "shot_continuity_report.html", output / "Shot_Continuity_Report.md"
    csv_path = output / "drift_timeline.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot(report, png_path)
    timeline = report.get("drift_timeline", [])
    if timeline:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timeline[0]))
            writer.writeheader(); writer.writerows(timeline)
    else:
        csv_path.write_text("from_frame,to_frame\n", encoding="utf-8")
    health = report["sequence_health"]
    rows = "".join(f"<tr><td>{html.escape(i['type'])}</td><td>{i['start_frame']} → {i['end_frame']}</td><td class='{i['severity']}'>{i['severity']}</td><td><code>{html.escape(json.dumps(i.get('evidence') or dict(), ensure_ascii=False))}</code></td><td>{html.escape(i['notes'])}</td></tr>" for i in report["issues"])
    cards = "".join(f"<span class='card'><b>{html.escape(k.replace('_',' ').title())}</b><br>{v:.1f}/100</span>" for k, v in health.items())
    bad_text = ", ".join(f"{item['frame']} ({item['severity']})" for item in report.get("bad_frames", [])) or "None"
    recommendations = ", ".join(str(value) for value in report.get("recommended_frames_to_regenerate", [])) or "None"
    html_path.write_text(f"<!doctype html><meta charset='utf-8'><title>Shot Continuity Report</title><style>body{{font-family:system-ui;max-width:1100px;margin:2rem auto;color:#18243a}}img{{max-width:100%}}.card{{display:inline-block;padding:.7rem;margin:.2rem;background:#eef3fa;border-radius:8px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.5rem;border-bottom:1px solid #dde4ef;text-align:left;vertical-align:top}}code{{font-size:.8rem;white-space:pre-wrap}}.High{{color:#b42318;font-weight:bold}}.Medium{{color:#a15c00;font-weight:bold}}</style><h1>Visual Continuity Lab — Shot Continuity Report</h1><p>Sequence: {html.escape(str(report['sequence']['name']))} · Frames: {report['sequence']['frame_count']}</p><p>{cards}</p><img src='{png_path.name}' alt='drift timeline'><h2>Bad Frames</h2><p>{html.escape(bad_text)}</p><h2>Recommended Frames to Regenerate</h2><p>{html.escape(recommendations)}</p><h2>Issues ({len(report['issues'])})</h2><table><tr><th>Type</th><th>Frames</th><th>Severity</th><th>Evidence</th><th>Notes</th></tr>{rows or '<tr><td colspan=5>None</td></tr>'}</table><h2>Demo markers</h2><pre>{html.escape(json.dumps(report.get('demo_markers', {}), indent=2))}</pre>", encoding="utf-8")
    md_path.write_text("# Shot Continuity Report\n\n" + "\n".join(f"- **{key.replace('_',' ').title()}**: {value:.2f}/100" for key, value in health.items()) + "\n\n## Bad Frames\n\n" + bad_text + "\n\n## Recommended Frames to Regenerate\n\n" + recommendations + "\n\n## Issues\n\n" + ("\n".join(f"- **{i['severity']}** `{i['type']}` — frames {i['start_frame']} → {i['end_frame']}: {i['notes']}" for i in report["issues"]) or "None") + "\n", encoding="utf-8")
    summary = {"input": report["sequence"].get("sequence_id"), "output": str(output), "issues": len(report["issues"]), "health": health}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"json": str(json_path), "html": str(html_path), "png": str(png_path), "markdown": str(md_path), "csv": str(csv_path)}
