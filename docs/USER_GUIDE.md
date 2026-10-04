# User guide

## 1. Start a project

Launch `VisualContinuityLab.exe` (or run the Python launcher) and create a
project. Give it a project name, then add one source:

- one image for a one-frame baseline;
- an image folder for a numbered sequence;
- a frame sequence/shot manifest;
- a video, which is sampled into a non-destructive frame sequence.

The application records a stable `Sequence ID`, `Shot`, `Frame ID`, frame
number, timestamp, and source path. Sorting by filename is only used to create
the initial order; the manifest is the source of truth afterward.

## 2. Lock the character region

Use the ROI tool to drag a rectangle around the character and choose **Lock
Character Region**. The lock is stored in normalized coordinates, so analysis
remains reproducible if the source images have different pixel dimensions.

If a shot contains more than one character, create one region per character.
Do not compare a region that contains a different subject in later frames;
mark that frame as an issue instead.

## 3. Choose a master reference

Select a clean frame and set **Master Reference**. The report then shows two
distinct views:

- **Adjacent consistency:** change from the previous frame to the current one;
- **Master consistency:** change from the selected master to the current one.

Adjacent drift is useful for finding sudden jumps. Master drift is useful for
finding slow accumulation.

## 4. Run analysis

Run **Analyze Sequence**. The engine computes basic measurements first, then
character/continuity dimensions, then issues and health summaries. A result is
ready only when the run has a completed status and the output manifest was
written.

The timeline lets you click a spike. The evidence panel should show the exact
frame pair, metric, measured values, threshold, and a visual comparison.

## 5. Review and label

Use Side by Side, Overlay, Difference, Slider, Blink, or Onion Skin to inspect a
flagged pair. Add a manual label (`Good`, `Bad Face`, `Bad Hair`, `Bad Costume`,
`Bad Pose`, or `Bad Background`) and optional notes. Labels are exported with
the frame ID and can later seed a project-specific QC dataset.

## 6. Regenerate recommendations

The report groups contiguous bad frames into issue ranges and recommends the
smallest ranges that cover high-severity events. This is a recommendation, not
an automatic deletion or overwrite of source images.

## 7. Export

Export the machine-readable JSON first. Then export the human-readable Shot
Continuity Report. JSON can be exchanged with FrameForge using the schema in
`schemas/frameforge_interchange.schema.json`; no FrameForge installation is
needed to analyze a sequence.

## Deterministic demo

Run the demo command to create a synthetic sequence with deliberate defects:

```powershell
python -m visual_continuity_lab --demo --output artifacts/demo
```

Or, after an editable install, use `vclab-demo analyze --input
demo/mock_sequence --output artifacts/demo`.

Expected findings include a colour change, a character-scale jump, a
brightness flicker, and a background change. The exact frame numbers are
recorded in the generated manifest, so tests should assert against the manifest
rather than hard-code assumptions about image ordering.
