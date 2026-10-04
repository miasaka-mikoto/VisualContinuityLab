"""Cross-module quality checks for the Visual Continuity Lab.

The tests in this package intentionally exercise the public seams (mock
sequence generator, sequence loader, analyser and report serialization).  They
do not reach into UI implementation details, so they remain useful for both
the Qt desktop build and headless CI.
"""

