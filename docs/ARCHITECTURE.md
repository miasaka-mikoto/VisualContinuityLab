# Architecture

Visual Continuity Lab is split into an engine, an adapter layer, and a UI.
The engine is usable without a desktop display so the same code can power the
CLI, tests, batch jobs, and the Windows executable.

```text
Input adapter
  image | folder | frame sequence | shot | video
        ↓
Sequence manifest (stable frame IDs, timestamps, paths)
        ↓
Measurement layer
  pixels · histogram · pHash · edges · features · flow
        ↓
ROI / reference layer
  locked character region + master reference
        ↓
Continuity analyzers
  face · hair · costume · body · position · scale · camera · background · lighting
        ↓
Drift events + evidence + health dimensions
        ↓
timeline / compare viewer / JSON + report export
```

## Boundaries

### Domain models

Sequence, Shot, Frame, CharacterRegion, Measurement, DriftEvent, Issue,
HealthSummary, and ManualLabel are plain serializable records. Their IDs are
stable across re-analysis so a review tool can attach notes without relying on
array positions.

### Ingestion

An ingestion adapter turns each supported source into a sequence manifest. A
video adapter samples frames and records both frame number and timestamp. The
original file path is retained as metadata; source pixels are read-only.

### Measurements

Measurements are intentionally low-level and reusable. Typical values include:

- luminance mean/stddev and contrast
- RGB/HSV histograms and colour centroids
- perceptual hash distance
- edge density and silhouette mask statistics
- keypoint match count/inlier ratio when a feature detector is available
- optical-flow magnitude and direction summary

Missing optional measurements are represented as `null` plus a reason in
evidence; they are not silently converted to a passing score.

### Analyzers

Each analyzer consumes a pair (or a pair plus master reference) of measurement
records and emits a normalized drift value, severity, and evidence. Thresholds
are configuration data. Adjacent and master comparisons are separate passes:

```text
Frame[n-1] ↔ Frame[n]       → adjacent continuity
Master reference ↔ Frame[n] → master continuity
```

The default severity bands are `low`, `medium`, and `high`; projects may tune
the numeric boundaries. A detector must identify which metric and threshold
caused an event so the UI can explain it.

### UI and exports

The UI is a thin consumer of engine results. It should never reimplement image
analysis or mutate source files. JSON exports are versioned. PDF/HTML/text
reports are projections of the same result model, making a CLI run and a GUI
run comparable.

## Health model

The report exposes separate percentage dimensions (0 means unstable and 100
means stable):

`identity_stability`, `motion_stability`, `color_stability`,
`lighting_stability`, `background_stability`, and `overall`.

`overall` is a weighted summary only; it must never replace the component
dimensions. A dimension with insufficient evidence is marked unavailable rather
than presented as a confident zero or one.

## Reference and ROI semantics

- A **master reference** is an explicit frame ID, not necessarily frame 1.
- A **locked character ROI** is stored in normalized coordinates (`x`, `y`,
  `width`, `height` in `[0, 1]`) plus the source image size used when it was
  selected.
- Pixel coordinates are derived at analysis time, so the same project remains
  portable between resolutions.
- Automatic person/face segmentation is an extension point; the locked ROI is
  the deterministic baseline.

## Extension points

1. Add a new analyzer implementing the shared result contract.
2. Add an ingestion adapter that yields the sequence manifest.
3. Add a report renderer without changing engine records.
4. Add a FrameForge-compatible field under the versioned interchange schema.

No extension should import FrameForge source code or require a network model.
