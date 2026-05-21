"""Tests for the report_cli module."""

import json
from pathlib import Path

import pytest

from sar_pattern_validation.report_cli import _load_workflow_result, main


@pytest.fixture()
def workflow_results_json(tmp_path: Path) -> Path:
    """Create a sample workflow results JSON file."""
    data = {
        "pass_rate_percent": 97.5,
        "evaluated_pixel_count": 1000,
        "passed_pixel_count": 975,
        "failed_pixel_count": 25,
        "gamma_image_path": None,
        "failure_image_path": None,
        "registered_overlay_path": None,
        "loaded_images_path": None,
        "reference_image_path": None,
        "measured_image_path": None,
        "aligned_measured_path": None,
        "measured_peak_wkg": 0.246,
        "measured_pssar": 24.63,
        "reference_pssar": 24.71,
        "scaling_error": -0.0035,
        "dose_to_agreement": 5.0,
        "distance_to_agreement": 2.0,
        "min_inscribed_square_mm": 22.0,
        "mask_fits_min_inscribed_square": True,
        "issues": [],
    }
    path = tmp_path / "workflow_results.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_workflow_result(workflow_results_json: Path):
    result = _load_workflow_result(str(workflow_results_json))
    assert result.pass_rate_percent == 97.5
    assert result.measured_pssar == 24.63
    assert result.gamma_image_path is None


def test_load_workflow_result_with_paths(tmp_path: Path):
    data = {
        "pass_rate_percent": 100.0,
        "evaluated_pixel_count": 500,
        "passed_pixel_count": 500,
        "failed_pixel_count": 0,
        "gamma_image_path": "/tmp/gamma.png",
        "failure_image_path": "None",
        "registered_overlay_path": None,
        "loaded_images_path": None,
        "reference_image_path": "/tmp/ref.png",
        "measured_image_path": None,
        "aligned_measured_path": None,
        "measured_peak_wkg": 0.5,
        "measured_pssar": 30.0,
        "reference_pssar": 30.0,
        "scaling_error": 0.0,
        "dose_to_agreement": 5.0,
        "distance_to_agreement": 2.0,
        "min_inscribed_square_mm": 22.0,
        "mask_fits_min_inscribed_square": True,
        "issues": [],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    result = _load_workflow_result(str(path))
    assert result.gamma_image_path == Path("/tmp/gamma.png")
    assert result.failure_image_path is None  # "None" string → None
    assert result.reference_image_path == Path("/tmp/ref.png")


def test_main_generates_report(tmp_path: Path, workflow_results_json: Path, capsys):
    """main() generates a report .tex file and prints success JSON."""
    from sar_pattern_validation.report import DEFAULT_TEMPLATE_DIR

    output_dir = tmp_path / "report_out"

    exit_code = main(
        [
            "--workflow-results-json",
            str(workflow_results_json),
            "--measured-file-path",
            "data/measurements/D900_Flat_HSL_15mm_10dBm_10g_3.csv",
            "--reference-file-path",
            "data/database/dipole_900MHz_Flat_15mm_10g.csv",
            "--power-level-dbm",
            "10.0",
            "--noise-floor",
            "0.001",
            "--report-output-dir",
            str(output_dir),
            "--report-template-dir",
            str(DEFAULT_TEMPLATE_DIR),
            "--antenna-type",
            "dipole",
            "--frequency-mhz",
            "900",
            "--distance-mm",
            "15",
            "--mass-g",
            "10",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "success"
    assert "report_path" in payload

    # Verify the .tex file was created
    tex_path = output_dir / "main.tex"
    assert tex_path.is_file()
    text = tex_path.read_text(encoding="utf-8")
    assert "dipole" in text
    assert "900" in text


def test_main_appends_to_existing_report(
    tmp_path: Path, workflow_results_json: Path, capsys
):
    """Calling main() twice appends a second page."""
    from sar_pattern_validation.report import DEFAULT_TEMPLATE_DIR

    output_dir = tmp_path / "report_out"

    base_args = [
        "--workflow-results-json",
        str(workflow_results_json),
        "--measured-file-path",
        "data/measurements/measured.csv",
        "--report-output-dir",
        str(output_dir),
        "--report-template-dir",
        str(DEFAULT_TEMPLATE_DIR),
    ]

    # First call
    main([*base_args, "--antenna-type", "dipole", "--frequency-mhz", "900"])
    capsys.readouterr()

    # Second call
    exit_code = main(
        [
            *base_args,
            "--antenna-type",
            "patch",
            "--frequency-mhz",
            "2450",
        ]
    )
    assert exit_code == 0

    tex_path = output_dir / "main.tex"
    text = tex_path.read_text(encoding="utf-8")
    assert "dipole" in text
    assert "patch" in text
    assert text.count(r"\end{document}") == 1

    # Two case figure directories
    assert (output_dir / "figures" / "case_000").is_dir()
    assert (output_dir / "figures" / "case_001").is_dir()


def test_main_returns_error_on_missing_json(tmp_path: Path, capsys):
    """main() returns error JSON when workflow results file is missing."""
    exit_code = main(
        [
            "--workflow-results-json",
            str(tmp_path / "nonexistent.json"),
            "--measured-file-path",
            "measured.csv",
        ]
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "error"
    assert "error" in payload
