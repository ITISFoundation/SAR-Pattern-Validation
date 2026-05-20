from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD_SCRIPT = (
    _REPO_ROOT
    / "scripts"
    / "measurement_validation"
    / "generate_measurement_validation_report_html.py"
)


def _load_dashboard_module():
    """Load generate_measurement_validation_report_html.py as a module.

    The script lives at the repo root and is meant to be executable. We import
    it dynamically so the dashboard generator's helpers are testable without
    invoking the CLI.
    """
    if "generate_measurement_validation_report_html" in sys.modules:
        return sys.modules["generate_measurement_validation_report_html"]
    spec = importlib.util.spec_from_file_location(
        "generate_measurement_validation_report_html", _DASHBOARD_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_measurement_validation_report_html"] = module
    spec.loader.exec_module(module)
    return module


def _make_case(
    *,
    case_id: str,
    pass_rate_percent: float | None,
    scaling_error: float | None,
    status: str = "passed",
    failed_pixel_count: int = 0,
    evaluated_pixel_count: int = 100,
    frequency_mhz: int = 900,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "frequency_key": f"{frequency_mhz}mhz",
        "frequency_label": f"{frequency_mhz} MHz",
        "frequency_mhz": frequency_mhz,
        "power_level_dbm": 10.0,
        "power_level_key": "10dbm",
        "distance_mm": 15,
        "averaging_mass": "10g",
        "evaluated_pixel_count": evaluated_pixel_count,
        "failed_pixel_count": failed_pixel_count,
        "passed_pixel_count": evaluated_pixel_count - failed_pixel_count,
        "pass_rate_percent": pass_rate_percent,
        "scaling_error": scaling_error,
        "status": status,
        "_report_name": "test_report.json",
    }


@pytest.fixture
def dashboard():
    return _load_dashboard_module()


class TestCombinedVerdict:
    def test_both_criteria_pass_yields_passed(self, dashboard) -> None:
        case = _make_case(case_id="ok", pass_rate_percent=100.0, scaling_error=0.05)
        assert (
            dashboard._combined_verdict(
                case, scaling_threshold_pct=10.0, gamma_threshold_pct=100.0
            )
            == "passed"
        )

    def test_gamma_below_threshold_yields_failed(self, dashboard) -> None:
        case = _make_case(case_id="g", pass_rate_percent=99.0, scaling_error=0.05)
        assert (
            dashboard._combined_verdict(
                case, scaling_threshold_pct=10.0, gamma_threshold_pct=100.0
            )
            == "failed"
        )

    def test_scaling_above_threshold_yields_failed(self, dashboard) -> None:
        case = _make_case(case_id="s", pass_rate_percent=100.0, scaling_error=0.15)
        assert (
            dashboard._combined_verdict(
                case, scaling_threshold_pct=10.0, gamma_threshold_pct=100.0
            )
            == "failed"
        )

    def test_error_status_yields_error(self, dashboard) -> None:
        case = _make_case(
            case_id="e",
            pass_rate_percent=None,
            scaling_error=None,
            status="error",
        )
        assert (
            dashboard._combined_verdict(
                case, scaling_threshold_pct=10.0, gamma_threshold_pct=100.0
            )
            == "error"
        )

    def test_missing_scaling_error_yields_error(self, dashboard) -> None:
        case = _make_case(case_id="m", pass_rate_percent=100.0, scaling_error=None)
        assert (
            dashboard._combined_verdict(
                case, scaling_threshold_pct=10.0, gamma_threshold_pct=100.0
            )
            == "error"
        )

    def test_threshold_is_configurable(self, dashboard) -> None:
        case_under = _make_case(
            case_id="under", pass_rate_percent=100.0, scaling_error=0.0499
        )
        case_over = _make_case(
            case_id="over", pass_rate_percent=100.0, scaling_error=0.0501
        )
        assert (
            dashboard._combined_verdict(
                case_under, scaling_threshold_pct=5.0, gamma_threshold_pct=100.0
            )
            == "passed"
        )
        assert (
            dashboard._combined_verdict(
                case_over, scaling_threshold_pct=5.0, gamma_threshold_pct=100.0
            )
            == "failed"
        )


class TestDashboardRendering:
    @pytest.mark.parametrize(
        ("pass_rate", "scaling_error", "expected_combined_badge"),
        [
            (100.0, 0.05, "passed"),
            (99.0, 0.05, "failed"),
            (100.0, 0.15, "failed"),
            (80.0, 0.20, "failed"),
        ],
    )
    def test_dashboard_renders_combined_verdict(
        self,
        dashboard,
        pass_rate: float,
        scaling_error: float,
        expected_combined_badge: str,
    ) -> None:
        case = _make_case(
            case_id="case_1",
            pass_rate_percent=pass_rate,
            scaling_error=scaling_error,
            status="passed" if pass_rate == 100.0 else "failed",
            failed_pixel_count=0 if pass_rate == 100.0 else 10,
        )
        html = dashboard._generate_html(
            reports=[{"name": "test_report.json"}],
            cases=[case],
            scaling_threshold_pct=10.0,
            gamma_threshold_pct=100.0,
        )
        # The data-combined attribute on the row reflects the verdict.
        assert f"data-combined='{expected_combined_badge}'" in html
        # The combined-pass summary card is present.
        assert "Combined Pass" in html

    def test_dashboard_respects_configurable_threshold(self, dashboard) -> None:
        case_under = _make_case(
            case_id="under",
            pass_rate_percent=100.0,
            scaling_error=0.0499,
            status="passed",
        )
        case_over = _make_case(
            case_id="over",
            pass_rate_percent=100.0,
            scaling_error=0.0501,
            status="passed",
        )
        # Default threshold (10%): both pass combined.
        html_default = dashboard._generate_html(
            reports=[{"name": "report.json"}],
            cases=[case_under, case_over],
        )
        assert "data-combined='passed'" in html_default
        # Both rows should be combined='passed' under default 10%.
        assert html_default.count("data-combined='passed'") == 2

        # Tighter threshold (5%): only the under-threshold case passes combined.
        html_tight = dashboard._generate_html(
            reports=[{"name": "report.json"}],
            cases=[case_under, case_over],
            scaling_threshold_pct=5.0,
        )
        assert html_tight.count("data-combined='passed'") == 1
        assert html_tight.count("data-combined='failed'") == 1

    def test_dashboard_threshold_label_in_summary_card(self, dashboard) -> None:
        case = _make_case(case_id="c", pass_rate_percent=100.0, scaling_error=0.05)
        html = dashboard._generate_html(
            reports=[{"name": "report.json"}],
            cases=[case],
            scaling_threshold_pct=7.5,
            gamma_threshold_pct=99.5,
        )
        assert "scaling ≤ ±7.5%" in html
        assert "gamma ≥ 99.5%" in html
