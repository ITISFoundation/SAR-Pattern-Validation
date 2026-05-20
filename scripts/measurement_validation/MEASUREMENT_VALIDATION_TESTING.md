# Measurement Validation Test Suite

This document describes the measurement validation testing infrastructure and how to use it.

## Overview

The measurement validation test suite validates SAR (Specific Absorption Rate) pattern measurements against reference data using gamma-map analysis. Tests are organized by frequency for easier reporting and analysis.

## Test Organization

Tests are automatically grouped by frequency:
- `test_measurement_workflow_cases_900mhz_match_reference_artifacts`
- `test_measurement_workflow_cases_1950mhz_match_reference_artifacts`
- `test_measurement_workflow_cases_2450mhz_match_reference_artifacts`
- `test_measurement_workflow_cases_5800mhz_match_reference_artifacts`

Each frequency group contains multiple cases varying by:
- Distance (e.g., 10 mm, 15 mm, 25 mm)
- Averaging mass (1g, 10g)
- Power level (e.g., 1 dBm, 4 dBm, 10 dBm, 17 dBm, 20 dBm)

## Quick Usage

### Run tests for missing cases only (smart rerun)
```bash
python scripts/measurement_validation/run_measurement_validation_tests.py
```
This will:
- Check which frequencies already have test artifacts
- Run only the tests for which artifacts are missing
- Generate a JSON report with aggregated results

### Force rerun specific frequencies
```bash
python scripts/measurement_validation/run_measurement_validation_tests.py --rerun 5800mhz --rerun 2450mhz
```
This will run all 5800 MHz and 2450 MHz cases, regardless of existing artifacts.

### Regenerate all artifacts
```bash
python scripts/measurement_validation/run_measurement_validation_tests.py --regenerate-artifacts
```
This will recompute all test cases and overwrite existing artifacts.

### Save visualization plots
```bash
python scripts/measurement_validation/run_measurement_validation_tests.py --save-plots
```
This will save diagnostic plots (loader comparison, registration overlay, gamma maps) for each test case.

### Combine options
```bash
python scripts/measurement_validation/run_measurement_validation_tests.py --rerun 5800mhz --save-plots
```

## Output Structure

### Artifacts Organization
Artifacts and logs are now organized by frequency:
```
tests/artifacts/measurement_validation/
├── 900mhz/
│   ├── case_id_metrics.json
│   └── case_id_gamma_field.npz
├── 1950mhz/
│   ├── case_id_metrics.json
│   └── case_id_gamma_field.npz
├── 2450mhz/
│   ├── case_id_metrics.json
│   └── case_id_gamma_field.npz
├── 5800mhz/
│   ├── case_id_metrics.json
│   └── case_id_gamma_field.npz
├── logs/
│   ├── 900mhz/
│   │   └── TIMESTAMP_testname_caseid.log
│   ├── 1950mhz/
│   │   └── ...
│   └── ...
└── plots/
    ├── 900mhz/
    │   └── caseid/
    │       ├── 01_loader_comparison.png
    │       ├── 02_registered_measured.png
    │       ├── 02_registration_overlay.png
    │       └── 03_gamma_map.png
    └── ...
```

### JSON Report
The test runner generates `tests/artifacts/measurement_validation/measurement_validation_report.json` with:
- Overall summary (total cases, pass/fail counts, aggregate pass rate)
- Per-frequency summaries and case details
- Run metadata (timestamp, duration, command-line args, platform info)

Example structure:
```json
{
  "schema_version": 1,
  "generated_at": "2026-04-09T...",
  "summary": {
    "case_count": 127,
    "passed_case_count": 111,
    "failed_case_count": 16,
    "frequency_count": 4,
    "aggregate_pass_rate_percent": 95.2
  },
  "frequencies": [
    {
      "frequency_key": "900mhz",
      "frequency_label": "900 MHz",
      "frequency_mhz": 900,
      "summary": {...},
      "case_ids": [...]
    }
  ],
  "cases": [
    {
      "case_id": "900_15mm_1g_10dbm_11",
      "status": "passed",
      "failed_pixel_count": 0,
      "pass_rate_percent": 100.0,
      ...
    }
  ],
  "run": {...}
}
```

## Generate HTML Dashboard

Generate the combined interactive dashboard from the per-frequency JSON reports:
```bash
python scripts/measurement_validation/generate_and_open_measurement_validation_dashboard.py --no-open
```

Or run the HTML generator directly:
```bash
python scripts/measurement_validation/generate_measurement_validation_report_html.py \
  --input-glob tests/artifacts/measurement_validation/reports/measurement_validation_report_*mhz.json \
  --output tests/artifacts/measurement_validation/reports/measurement_validation_dashboard.html
```

The dashboard is written to `tests/artifacts/measurement_validation/reports/measurement_validation_dashboard.html`.

The HTML dashboard shows:
- Summary cards with pass/fail statistics
- Color-coded results by frequency
- Detailed case tables with failure details
- Run metadata and metrics

Note: older notes may still refer to `tests/artifacts/measurement_validation/report.html`, but the current workflow writes the combined dashboard under `tests/artifacts/measurement_validation/reports/`.

## Case Naming

Cases follow the pattern: `{frequency}_{distance}mm_{mass}_{power}dbm_{index}`

Examples:
- `900_15mm_1g_10dbm_11` → 900 MHz, 15 mm distance, 1g averaging, 10 dBm power, index 11
- `2450_10mm_1g_17dbm_5` → 2450 MHz, 10 mm distance, 1g averaging, 17 dBm power, index 5
- `5800_10mm_10g_1dbm_22` → 5.8 GHz, 10 mm distance, 10g averaging, 1 dBm power, index 22

## Direct pytest Usage

You can also run tests directly with pytest:

```bash
# Run all tests
pytest tests/test_measurement_validation.py -v

# Run tests for a specific frequency
pytest tests/test_measurement_validation.py::test_measurement_workflow_cases_900mhz_match_reference_artifacts -v

# Run with regeneration
REGENERATE_MEASUREMENT_VALIDATION_ARTIFACTS=1 pytest tests/test_measurement_validation.py

# Run with specific marker
pytest tests/test_measurement_validation.py -m slow
```

If `uv run pytest ...` cannot use the local `.venv`, you can run the same tests through an isolated `uvx` environment with this package installed editable:

```bash
# Reusable fallback for this repo. pytest-xdist is included because pyproject.toml
# adds `-n auto --dist worksteal` through pytest addopts.
uvx --isolated --python 3.12 --from pytest --with pytest-xdist --with-editable . \
  pytest -q tests/test_notebook_boundary.py

uvx --isolated --python 3.12 --from pytest --with pytest-xdist --with-editable . \
  pytest -q tests/test_sample_catalog_cli.py
```

## Notes

- Test cases are marked with `@pytest.mark.slow` and use xdist for parallel execution
- By default, only "missing" cases run (those without existing artifacts)
- Use `--regenerate-artifacts` to force recalculation
- Artifacts are organized by frequency for cleaner management
- Old artifacts with "zip_" prefix are automatically detected and migrated
- The JSON report is designed for both programmatic consumption and HTML visualization
