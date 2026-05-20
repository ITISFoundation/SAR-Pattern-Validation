#!/usr/bin/env python
"""Build a single measurement-validation dashboard HTML and optionally open it."""

from __future__ import annotations

import argparse
import subprocess  # nosec B404
import sys
import webbrowser
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a combined measurement-validation HTML dashboard and open it."
    )
    parser.add_argument(
        "--input-glob",
        default="tests/artifacts/measurement_validation/reports/measurement_validation_report_*mhz.json",
        help="Glob for input JSON report files.",
    )
    parser.add_argument(
        "--output",
        default="tests/artifacts/measurement_validation/reports/measurement_validation_dashboard.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Generate only; do not open in browser.",
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    output_path = (repo_root / args.output).resolve()

    cmd = [
        sys.executable,
        str(script_dir / "generate_measurement_validation_report_html.py"),
        "--input-glob",
        args.input_glob,
        "--output",
        str(output_path),
    ]

    proc = subprocess.run(cmd, cwd=repo_root)  # nosec B603
    if proc.returncode != 0:
        return proc.returncode

    if not args.no_open:
        webbrowser.open(output_path.as_uri())
        print(f"Opened dashboard: {output_path}")
    else:
        print(f"Dashboard generated: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
