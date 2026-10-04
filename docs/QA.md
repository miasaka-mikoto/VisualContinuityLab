# QA and verification

The project is considered usable only when the deterministic demo, unit tests,
and a small real sequence all complete.

## Required checks

```powershell
python -m pytest -q
python -m visual_continuity_lab --demo --output artifacts/demo
```

The demo must produce:

- a sequence manifest with stable frame IDs;
- measurements for every frame;
- at least one issue for colour, scale, brightness, and background defects;
- adjacent and master continuity results;
- a timeline/health summary;
- machine-readable JSON and a human-readable report.

## Regression strategy

Use fixed seeds and generated fixtures for detector tests. Assert metric
relationships (for example, the intentionally bright frame has higher
brightness drift) and issue frame IDs from the manifest. Avoid asserting exact
floating-point values unless rounded by the public result contract.

When adding a detector, include one positive synthetic fixture, one negative
fixture, and a missing-data case. The missing-data case should verify that the
report explains why a dimension is unavailable.

