# Verification screenshots

The deterministic demo renderer writes `artifacts/demo_analysis/drift_report.png`
after:

```powershell
python scripts/analyze_demo.py run --demo demo/mock_sequence --output artifacts/demo_analysis
```

The image is a real drift timeline (not a UI placeholder): colour, character
scale, brightness, and background signals are plotted across the generated
frames. The same folder contains the JSON, CSV, HTML, and Markdown report used
to cross-check the visual plot.

