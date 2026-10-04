"""Portable JSON and FrameForge interchange helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from .models import ContinuityReport, DriftDimension, DriftResult, FrameRecord, HealthScores, Issue, ROI, Sequence

FRAMEFORGE_SCHEMA_VERSION = "1.0"
APPLICATION_VERSION = "0.1.0"


def _read_payload(source: str | Path | bytes | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    if isinstance(source, bytes):
        return dict(json.loads(source))
    value = str(source)
    try:
        path = Path(value)
        if path.exists() and path.is_file():
            return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        # A JSON document can be much longer than a filesystem path.
        pass
    return dict(json.loads(value))


def _comparison(result: DriftResult | None, frame_number: int = 0) -> dict[str, Any] | None:
    if result is None:
        return None
    metrics = {name: float(dim.score) for name, dim in result.dimensions.items()}
    evidence = []
    for name, dim in result.dimensions.items():
        info = dict(dim.evidence or {})
        evidence.append({
            "metric": name,
            "value": float(dim.score),
            "referenceValue": info.get("from_value", info.get("from")),
            "threshold": None,
            "message": f"{name} drift is {dim.severity} (score {dim.score:.3f}).",
            "details": info,
        })
    return {
        "overallDrift": float(result.overall_score),
        "severity": result.severity,
        "metrics": metrics,
        "evidence": evidence,
    }


def export_frameforge(
    sequence: Sequence,
    report: ContinuityReport | None = None,
    *,
    project_id: str | None = None,
    application_version: str = APPLICATION_VERSION,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Export a sequence/report to the versioned FrameForge JSON contract."""

    adjacent_by_to = {r.to_frame: r for r in (report.adjacent_results if report else [])}
    master_by_to = {r.to_frame: r for r in (report.master_results if report else [])}
    feature_map = report.frame_features if report else {}
    issue_ids_by_frame: dict[str, list[str]] = {}
    if report:
        for issue in report.issues:
            issue_ids_by_frame.setdefault(issue.start_frame, []).append(issue.issue_id)
            issue_ids_by_frame.setdefault(issue.end_frame, []).append(issue.issue_id)
    frames = []
    for frame in sequence.frames:
        feature = feature_map.get(frame.frame_id)
        measurements: dict[str, Any] = {}
        if feature:
            measurements = {
                "imageWidth": feature.image_size[0], "imageHeight": feature.image_size[1],
                "brightness": feature.brightness, "contrast": feature.contrast,
                "edgeDensity": feature.edge_density, "perceptualHash": feature.perceptual_hash,
                "colorDistribution": feature.color_distribution,
            }
        status_value = {"analysed": "analyzed", "analyzed": "analyzed", "error": "failed",
                        "failed": "failed", "unanalysed": "pending", "pending": "pending",
                        "skipped": "skipped", "labeled": "labeled", "generated": "analyzed"}.get(frame.status, "pending")
        frame_payload: dict[str, Any] = {
            "frameId": frame.frame_id,
            "frameNumber": int(frame.frame_number),
            "character": frame.character or sequence.character,
            "shot": frame.shot or sequence.shot,
            "status": status_value,
            "measurements": measurements,
            "continuity": {
                "adjacent": _comparison(adjacent_by_to.get(frame.frame_id)),
                "master": _comparison(master_by_to.get(frame.frame_id)),
            },
            "labels": [label for label in sequence.manual_labels if label.get("frame_id") == frame.frame_id],
            "issueIds": sorted(set(issue_ids_by_frame.get(frame.frame_id, []))),
            "notes": frame.metadata.get("notes", ""),
        }
        if frame.timestamp is not None:
            frame_payload["timestamp"] = float(frame.timestamp)
        if frame.path:
            frame_payload["path"] = frame.path
        if frame.roi:
            frame_payload["roi"] = frame.roi.to_dict()
        frames.append(frame_payload)
    issues = []
    if report:
        index = {f.frame_id: f.frame_number for f in sequence.frames}
        for issue in report.issues:
            issues.append({
                "issueId": issue.issue_id, "type": issue.type,
                "startFrame": int(index.get(issue.start_frame, 0)),
                "endFrame": int(index.get(issue.end_frame, 0)),
                "severity": issue.severity,
                "evidence": [
                    {"metric": issue.type, "message": issue.notes, "details": issue.evidence}
                ],
                "notes": issue.notes,
                "recommendedAction": "Regenerate or review the flagged frames.",
            })
    health = None
    if report:
        health = {
            "identityStability": report.health.identity_stability,
            "motionStability": report.health.motion_stability,
            "colorStability": report.health.color_stability,
            "lightingStability": report.health.lighting_stability,
            "backgroundStability": report.health.background_stability,
            "overall": report.health.overall,
        }
    payload = {
        "schemaVersion": FRAMEFORGE_SCHEMA_VERSION,
        "projectId": project_id or sequence.sequence_id,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "source": {"application": "Visual Continuity Lab", "version": application_version,
                   "sequenceId": sequence.sequence_id},
        "settings": {},
        "shots": [{"shotId": sequence.shot or sequence.sequence_id,
                   "name": sequence.shot or sequence.sequence_id,
                   "masterReferenceFrameId": sequence.master_reference,
                   "characters": ([{"characterId": sequence.character}] if sequence.character else []),
                   "frames": frames}],
        "issues": issues,
    }
    fps = sequence.metadata.get("fps")
    if fps is not None and float(fps) > 0:
        payload["shots"][0]["frameRate"] = float(fps)
    roi_payload = next((f.roi.to_dict() for f in sequence.frames if f.roi), None)
    if roi_payload:
        payload["settings"]["roi"] = roi_payload
    if health is not None:
        payload["health"] = health
    if output is not None:
        Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def export_frameforge_json(sequence: Sequence, report: ContinuityReport | None = None, **kwargs: Any) -> str:
    """Return FrameForge interchange as a JSON string."""
    return json.dumps(export_frameforge(sequence, report, **kwargs), ensure_ascii=False, indent=2)


def sequence_from_frameforge(source: str | Path | bytes | Mapping[str, Any]) -> Sequence:
    """Read the first shot from a FrameForge interchange payload."""
    data = _read_payload(source)
    shots = data.get("shots", [])
    shot = shots[0] if shots else data
    frames = []
    for item in shot.get("frames", data.get("frames", [])):
        roi_data = item.get("roi")
        frames.append(FrameRecord(
            frame_id=str(item.get("frameId", item.get("frame_id", ""))),
            path=item.get("path"), frame_number=int(item.get("frameNumber", item.get("frame_number", 0)) or 0),
            timestamp=(float(item["timestamp"]) if item.get("timestamp") is not None else None),
            character=item.get("character"), shot=item.get("shot", shot.get("shotId")),
            status={"analyzed": "analysed", "failed": "error", "pending": "unanalysed"}.get(str(item.get("status", "unanalysed")), str(item.get("status", "unanalysed"))),
            roi=ROI.from_dict(roi_data), metadata={"frameforge": item},
        ))
    sequence = Sequence(sequence_id=str(data.get("projectId", data.get("sequence_id", shot.get("shotId", "sequence")))),
                        frames=frames, master_reference=shot.get("masterReferenceFrameId"),
                        shot=shot.get("shotId"), character=(shot.get("characters") or [{}])[0].get("characterId"),
                        manual_labels=[label for item in shot.get("frames", []) for label in item.get("labels", [])])
    return sequence


def report_from_frameforge(source: str | Path | bytes | Mapping[str, Any]) -> ContinuityReport:
    """Reconstruct a report from per-frame FrameForge comparisons."""
    data = _read_payload(source)
    sequence = sequence_from_frameforge(data)
    shot = (data.get("shots") or [{}])[0]
    adjacent: list[DriftResult] = []
    master: list[DriftResult] = []
    frame_ids = [f.frame_id for f in sequence.frames]
    for index, item in enumerate(shot.get("frames", [])):
        fid = str(item.get("frameId", ""))
        for key, target in (("adjacent", adjacent), ("master", master)):
            cmp = (item.get("continuity") or {}).get(key)
            if not cmp:
                continue
            if key == "adjacent":
                from_id = frame_ids[index - 1] if index > 0 and index - 1 < len(frame_ids) else fid
            else:
                from_id = sequence.master_reference or fid
            dims = {
                name: DriftDimension(name, float(score), str(cmp.get("severity", "low")), evidence={})
                for name, score in (cmp.get("metrics") or {}).items()
            }
            target.append(DriftResult(from_id, fid, dims, float(cmp.get("overallDrift", 0.0)), str(cmp.get("severity", "low")), dict(cmp), key))
    issues = []
    for item in data.get("issues", []):
        start_num, end_num = int(item.get("startFrame", 0)), int(item.get("endFrame", 0))
        by_num = {f.frame_number: f.frame_id for f in sequence.frames}
        # The interchange schema intentionally stores evidence as an array of
        # explainable records.  The internal Issue model keeps a mapping so it
        # can be indexed cheaply; preserve the complete array under ``items``
        # instead of attempting ``dict(list)`` (which fails for the normal
        # ``[{metric, message, details}, ...]`` payload).
        raw_evidence = item.get("evidence", {})
        if isinstance(raw_evidence, Mapping):
            issue_evidence = dict(raw_evidence)
        elif isinstance(raw_evidence, list):
            issue_evidence = {"items": raw_evidence}
        else:
            issue_evidence = {"value": raw_evidence}
        issues.append(Issue(
            str(item.get("issueId", "")),
            str(item.get("type", "")),
            by_num.get(start_num, str(start_num)),
            by_num.get(end_num, str(end_num)),
            str(item.get("severity", "low")),
            issue_evidence,
            str(item.get("notes", "")),
        ))
    h = data.get("health") or {}
    health = HealthScores(float(h.get("identityStability", 100)), float(h.get("motionStability", 100)), float(h.get("colorStability", 100)), float(h.get("lightingStability", 100)), float(h.get("backgroundStability", 100)), float(h.get("overall", 100)), len(adjacent))
    return ContinuityReport(sequence.sequence_id, adjacent, master, issues, health=health,
                            bad_frames=[i.start_frame for i in issues], metadata={"schemaVersion": data.get("schemaVersion")})


def import_frameforge(source: str | Path | bytes | Mapping[str, Any]) -> tuple[Sequence, ContinuityReport | None]:
    """Import both sequence metadata and an optional report."""
    sequence = sequence_from_frameforge(source)
    data = _read_payload(source)
    has_analysis = bool(data.get("health") or data.get("issues"))
    return sequence, report_from_frameforge(data) if has_analysis else None


def export_json(obj: Sequence | ContinuityReport, output: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
    """Generic JSON export for a sequence or report."""
    payload = obj.to_dict() if isinstance(obj, (Sequence, ContinuityReport)) else dict(obj)
    if output is not None:
        Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def import_json(source: str | Path | bytes | Mapping[str, Any]) -> Sequence | ContinuityReport:
    data = _read_payload(source)
    if "shots" in data and "schemaVersion" in data:
        return sequence_from_frameforge(data)
    if "adjacent_results" in data or "master_results" in data or "health" in data and "sequence_id" in data:
        return ContinuityReport.from_dict(data)
    return Sequence.from_dict(data)


__all__ = [
    "APPLICATION_VERSION", "FRAMEFORGE_SCHEMA_VERSION", "export_frameforge",
    "export_frameforge_json", "export_json", "import_frameforge", "import_json",
    "report_from_frameforge", "sequence_from_frameforge",
]
