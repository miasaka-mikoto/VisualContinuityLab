#!/usr/bin/env python3
"""Generate/analyze the deterministic Visual Continuity Lab demo.

The command works directly from a source checkout and emits analysis JSON, an
HTML report, a PNG drift graph, and a short Markdown handoff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo_analysis import analyze_sequence, write_report  # noqa: E402
from scripts.mock_sequence import generate_mock_sequence  # noqa: E402


def _load_and_analyze(source: Path, master_index: int = 0) -> dict:
    """Backward-compatible adapter retained for early packaging scripts."""
    del master_index  # the manifest's master_reference is authoritative
    return analyze_sequence(source)


def generate(args: argparse.Namespace) -> int:
    manifest = generate_mock_sequence(args.output, frame_count=args.frames, width=args.width, height=args.height, fps=args.fps)
    print(json.dumps({"output": str(args.output), "sequence_id": manifest["sequence_id"], "frames": manifest["frame_count"], "markers": manifest["demo_markers"]}, indent=2, ensure_ascii=False))
    return 0


def analyze(args: argparse.Namespace) -> int:
    source = Path(args.input).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    report = analyze_sequence(source)
    paths = write_report(report, Path(args.output).expanduser().resolve())
    summary = {"input": str(source), "output": paths, "issues": len(report.get("issues", [])), "bad_frames": report.get("bad_frames", []), "health": report.get("sequence_health", {})}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def run(args: argparse.Namespace) -> int:
    generate_mock_sequence(args.demo, frame_count=args.frames, width=args.width, height=args.height, fps=args.fps)
    return analyze(argparse.Namespace(input=args.demo, output=args.output))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="create deterministic local PNG frames")
    generate_parser.add_argument("--output", type=Path, default=Path("demo/mock_sequence"))
    generate_parser.add_argument("--frames", type=int, default=24)
    generate_parser.add_argument("--width", type=int, default=640)
    generate_parser.add_argument("--height", type=int, default=360)
    generate_parser.add_argument("--fps", type=float, default=12.0)
    generate_parser.set_defaults(handler=generate)

    analyze_parser = subparsers.add_parser("analyze", help="analyze an image sequence and write reports")
    analyze_parser.add_argument("--input", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, default=Path("artifacts/demo_analysis"))
    analyze_parser.set_defaults(handler=analyze)

    run_parser = subparsers.add_parser("run", help="generate the demo, then analyze it")
    run_parser.add_argument("--demo", type=Path, default=Path("demo/mock_sequence"))
    run_parser.add_argument("--output", type=Path, default=Path("artifacts/demo_analysis"))
    run_parser.add_argument("--frames", type=int, default=24)
    run_parser.add_argument("--width", type=int, default=640)
    run_parser.add_argument("--height", type=int, default=360)
    run_parser.add_argument("--fps", type=float, default=12.0)
    run_parser.set_defaults(handler=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Preserve the early one-line CLI forms used by packaging smoke tests:
    # ``analyze_demo.py --input seq`` and a no-argument local demo run.
    commands = {"generate", "analyze", "run", "-h", "--help"}
    if not raw:
        raw = ["run"]
    elif raw[0] not in commands:
        raw = ["analyze" if "--input" in raw else "run", *raw]
    args = build_parser().parse_args(raw)
    return int(args.handler(args))


__all__ = ["analyze_sequence", "write_report", "generate_mock_sequence", "build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
