"""
CLI entry point for standalone report generation.

Generates (or appends to) the SAR Pattern Validation PDF report from
previously saved workflow results, without re-running the gamma workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from sar_pattern_validation.report import DEFAULT_TEMPLATE_DIR, generate_report
from sar_pattern_validation.workflow_config import WorkflowConfig
from sar_pattern_validation.workflows import WorkflowResult


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or append a page to the SAR validation report."
    )
    parser.add_argument(
        "--workflow-results-json",
        type=str,
        required=True,
        help="Path to the workflow results JSON file.",
    )
    parser.add_argument(
        "--measured-file-path",
        type=str,
        required=True,
        help="Path to the measured CSV (used for filename display in report).",
    )
    parser.add_argument(
        "--reference-file-path",
        type=str,
        default="reference.csv",
        help="Path to the reference CSV.",
    )
    parser.add_argument(
        "--power-level-dbm",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--measurement-area-x-mm",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--measurement-area-y-mm",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--report-output-dir",
        type=str,
        default="report",
        help="Output directory for the generated LaTeX report.",
    )
    parser.add_argument(
        "--report-template-dir",
        type=str,
        default=None,
        help="Override path to the LaTeX report template directory.",
    )
    parser.add_argument(
        "--antenna-type",
        type=str,
        default="dipole",
    )
    parser.add_argument(
        "--frequency-mhz",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--distance-mm",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--mass-g",
        type=int,
        default=0,
    )
    return parser


def _load_workflow_result(json_path: str) -> WorkflowResult:
    """Load WorkflowResult from a JSON file (as saved by workflow_cli)."""
    data: dict[str, Any] = json.loads(Path(json_path).read_text(encoding="utf-8"))

    # Convert string paths back to Path objects for image fields
    path_fields = {
        "gamma_image_path",
        "failure_image_path",
        "registered_overlay_path",
        "loaded_images_path",
        "reference_image_path",
        "measured_image_path",
        "aligned_measured_path",
    }
    for key in path_fields:
        val = data.get(key)
        if val is not None and val != "None":
            data[key] = Path(val)
        else:
            data[key] = None

    # Remove fields not in WorkflowResult dataclass (e.g. extra metadata)
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(WorkflowResult)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}

    return WorkflowResult(**filtered)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for report generation."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        workflow_result = _load_workflow_result(args.workflow_results_json)

        workflow_config = WorkflowConfig(
            measured_file_path=args.measured_file_path,
            reference_file_path=args.reference_file_path,
            power_level_dbm=args.power_level_dbm,
            noise_floor=args.noise_floor,
            measurement_area_x_mm=args.measurement_area_x_mm,
            measurement_area_y_mm=args.measurement_area_y_mm,
        )

        template_dir = (
            Path(args.report_template_dir)
            if args.report_template_dir
            else DEFAULT_TEMPLATE_DIR
        )

        report_path = generate_report(
            workflow_result=workflow_result,
            workflow_config=workflow_config,
            output_dir=args.report_output_dir,
            template_dir=template_dir,
            antenna_type=args.antenna_type,
            frequency_mhz=args.frequency_mhz,
            distance_mm=args.distance_mm,
            mass_g=args.mass_g,
        )

        payload = {
            "status": "success",
            "report_path": str(report_path),
        }
        print(json.dumps(payload, indent=2))
        return 0

    except Exception as exc:
        error_payload = {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        print(json.dumps(error_payload, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
