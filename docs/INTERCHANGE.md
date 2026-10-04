# FrameForge interchange

Visual Continuity Lab exchanges JSON only. The contract is deliberately small
and versioned so FrameForge can consume continuity results without importing
VCL modules.

## Root fields

`schemaVersion` identifies the contract. `projectId`, `source`, and `shots`
identify the export. A shot contains a master reference and ordered frames.
Each frame may carry a `continuity` result, manual labels, and issue IDs.

The schema is permissive about future extension keys, but stable fields have
explicit types and ranges. Producers should preserve unknown keys when
round-tripping an export.

## Minimal example

```json
{
  "schemaVersion": "1.0",
  "projectId": "vcl-demo",
  "source": {"application": "Visual Continuity Lab", "version": "0.1.0"},
  "shots": [{
    "shotId": "shot-001",
    "masterReferenceFrameId": "frame-001",
    "frames": [{
      "frameId": "frame-001",
      "frameNumber": 1,
      "timestamp": 0.0,
      "status": "analyzed",
      "continuity": {
        "adjacent": null,
        "master": {"overallDrift": 0.0, "severity": "low"}
      }
    }]
  }]
}
```

See `schemas/frameforge_interchange.schema.json` for the authoritative shape.

