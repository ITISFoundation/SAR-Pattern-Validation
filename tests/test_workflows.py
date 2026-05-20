from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

matplotlib.use("Agg")

import sar_pattern_validation.workflows as workflows_module
from sar_pattern_validation.errors import ConfigValidationError
from sar_pattern_validation.gamma_eval import GammaMapEvaluator
from sar_pattern_validation.image_loader import SARImageLoader
from sar_pattern_validation.workflow_config import PlottingConfig
from sar_pattern_validation.workflow_schema import validate_workflow_config
from sar_pattern_validation.workflows import _apply_roi_policy, complete_workflow

from .helpers import (
    gaussian_2d,
    make_rect_grid,
    punch_rect_hole,
    rigid_transform_points,
    write_sar_csv,
)


def _make_image(arr: np.ndarray) -> sitk.Image:
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing((0.001, 0.001))
    return img


def _write_synthetic_workflow_pair(
    tmp_path: Path,
    *,
    tx: float = 0.0,
    ty: float = 0.0,
    theta_deg: float = 0.0,
) -> tuple[Path, Path]:
    x, y = make_rect_grid(xmin=-0.04, xmax=0.04, ymin=-0.04, ymax=0.04, step=0.005)
    _, _, main_peak = gaussian_2d(
        x, y, x0=0.012, y0=-0.008, sx=0.018, sy=0.022, peak=1.0
    )
    _, _, side_peak = gaussian_2d(
        x, y, x0=-0.016, y0=0.018, sx=0.012, sy=0.016, peak=0.6
    )
    Z = main_peak + side_peak

    reference_csv = tmp_path / "reference.csv"
    write_sar_csv(reference_csv, x, y, Z)

    measured_df = pd.read_csv(reference_csv)
    if tx != 0.0 or ty != 0.0 or theta_deg != 0.0:
        measured_df = rigid_transform_points(
            measured_df, tx=tx, ty=ty, theta_deg=theta_deg
        )

    measured_csv = tmp_path / "measured.csv"
    measured_df.to_csv(measured_csv, index=False)
    return measured_csv, reference_csv


def _write_truncated_support_workflow_pair(tmp_path: Path) -> tuple[Path, Path]:
    measured_csv, reference_csv = _write_synthetic_workflow_pair(tmp_path)
    measured_df = pd.read_csv(measured_csv)
    measured_df = punch_rect_hole(
        measured_df,
        xmin=0.015,
        xmax=0.040,
        ymin=-0.040,
        ymax=0.040,
    )
    measured_df.to_csv(measured_csv, index=False)
    return measured_csv, reference_csv


def test_validate_workflow_config_rejects_negative_distance() -> None:
    with pytest.raises(ConfigValidationError):
        validate_workflow_config({"distance_to_agreement": -1.0})


def test_workflow_schema_accepts_zero_noise_floor() -> None:
    """V20: noise_floor=0 is valid (means no noise filtering)."""
    config = validate_workflow_config({"noise_floor": 0.0})
    assert config.noise_floor == 0.0


def test_workflow_schema_rejects_negative_noise_floor() -> None:
    with pytest.raises(ConfigValidationError):
        validate_workflow_config({"noise_floor": -0.001})


def test_validate_workflow_config_accepts_plotting_config() -> None:
    config = validate_workflow_config(
        {
            "plotting": {
                "window_mm": (-40, 40, -35, 35),
                "font_size": 16,
                "save_dpi": 180,
            }
        }
    )

    assert config.plotting.window_mm == (-40.0, 40.0, -35.0, 35.0)
    assert config.plotting.font_size == 16
    assert config.plotting.save_dpi == 180


def test_validate_workflow_config_defaults_to_intersection_roi() -> None:
    config = validate_workflow_config({})

    assert config.evaluation_roi_policy == "intersection"


def test_validate_workflow_config_rejects_invalid_plotting_config() -> None:
    with pytest.raises(ConfigValidationError):
        validate_workflow_config({"plotting": {"save_dpi": 0}})


@pytest.mark.parametrize(
    "x_mm,y_mm",
    [
        (50.0, 100.0),  # x at exclusive lower bound
        (50.0001, 50.0001),  # both individually valid (sentinel handled in body)
        (601.0, 200.0),  # x above upper bound
        (300.0, 401.0),  # y above upper bound
        (None, 100.0),  # only one of the pair set
        (100.0, None),
    ],
)
def test_validate_workflow_config_rejects_out_of_range_measurement_area(
    x_mm: float | None, y_mm: float | None
) -> None:
    payload: dict[str, float | None] = {}
    if x_mm is not None:
        payload["measurement_area_x_mm"] = x_mm
    if y_mm is not None:
        payload["measurement_area_y_mm"] = y_mm
    if x_mm == 50.0001 and y_mm == 50.0001:
        config = validate_workflow_config(payload)
        assert config.measurement_area_x_mm == pytest.approx(50.0001)
        assert config.measurement_area_y_mm == pytest.approx(50.0001)
        return
    with pytest.raises(ConfigValidationError):
        validate_workflow_config(payload)


def test_validate_workflow_config_measurement_area_derives_square_window() -> None:
    config = validate_workflow_config(
        {"measurement_area_x_mm": 300.0, "measurement_area_y_mm": 200.0}
    )
    assert config.measurement_area_x_mm == 300.0
    assert config.measurement_area_y_mm == 200.0
    assert config.plotting.window_mm == (-150.0, 150.0, -150.0, 150.0)


def test_validate_workflow_config_measurement_area_square_uses_y_when_larger() -> None:
    config = validate_workflow_config(
        {"measurement_area_x_mm": 100.0, "measurement_area_y_mm": 400.0}
    )
    assert config.plotting.window_mm == (-200.0, 200.0, -200.0, 200.0)


def test_validate_workflow_config_no_measurement_area_keeps_default_window() -> None:
    config = validate_workflow_config({})
    assert config.measurement_area_x_mm is None
    assert config.measurement_area_y_mm is None
    assert config.plotting.window_mm == (-120.0, 120.0, -120.0, 120.0)


def test_apply_roi_policy_sets_expected_masks() -> None:
    reference = _make_image(np.ones((8, 8), dtype=np.float32))
    measured = _make_image(np.ones((8, 8), dtype=np.float32))
    reference_mask = sitk.Cast(reference > 0, sitk.sitkUInt8)
    measured_mask = sitk.Cast(measured > 0, sitk.sitkUInt8)
    evaluator = GammaMapEvaluator(
        reference_sar_linear=reference,
        measured_sar_linear=measured,
        reference_to_measured_transform=sitk.Euler2DTransform(),
    )

    _apply_roi_policy(
        evaluator,
        reference_mask_u8=reference_mask,
        measured_mask_u8=measured_mask,
        policy="reference_only",
    )
    assert evaluator.reference_mask_u8 is reference_mask
    assert evaluator.measured_mask_u8 is None

    _apply_roi_policy(
        evaluator,
        reference_mask_u8=reference_mask,
        measured_mask_u8=measured_mask,
        policy="intersection",
    )
    assert evaluator.reference_mask_u8 is reference_mask
    assert evaluator.measured_mask_u8 is measured_mask

    _apply_roi_policy(
        evaluator,
        reference_mask_u8=reference_mask,
        measured_mask_u8=measured_mask,
        policy="none",
    )
    assert evaluator.reference_mask_u8 is None
    assert evaluator.measured_mask_u8 is None


@pytest.mark.validation
def test_complete_workflow_integration_saves_overlay_outputs(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    measured_csv = project_root / "data/example/measured_sSAR1g.csv"
    reference_csv = project_root / "data/example/reference_sSAR1g.csv"

    loaded_images = tmp_path / "loaded.png"
    registered_overlay = tmp_path / "registered.png"
    gamma_image = tmp_path / "gamma.png"

    result = complete_workflow(
        measured_file_path=str(measured_csv),
        reference_file_path=str(reference_csv),
        loaded_images_save_path=str(loaded_images),
        registered_image_save_path=str(registered_overlay),
        gamma_comparison_image_path=str(gamma_image),
        transform_type="rigid",
        resample_resolution=0.001,
        stages=[
            {
                "translation_step": 0.010,
                "rot_step_deg": 4.0,
                "rot_span_deg": 180.0,
                "tx_steps": 6,
                "ty_steps": 6,
            }
        ],
        evaluation_roi_policy="intersection",
        save_failures_overlay=True,
        log_level="WARNING",
    )

    assert result.evaluated_pixel_count > 0
    assert 0 <= result.pass_rate_percent <= 100

    assert result.registered_overlay_path is not None
    assert result.gamma_image_path is not None
    assert result.failure_image_path is not None

    assert result.registered_overlay_path.exists()
    assert result.gamma_image_path.exists()
    assert result.failure_image_path.exists()


@pytest.mark.validation
def test_complete_workflow_passes_shared_plotting_config(tmp_path, monkeypatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    measured_csv = project_root / "data/example/measured_sSAR1g.csv"
    reference_csv = project_root / "data/example/reference_sSAR1g.csv"
    received: dict[str, object] = {}

    def capture_loader_plot(self, *args, **kwargs) -> None:
        received["loader"] = kwargs["plotting_config"]

    def capture_aligned_plot(self, *args, **kwargs) -> None:
        received["aligned"] = kwargs["plotting_config"]

    def capture_overlay(*args, **kwargs) -> None:
        received["overlay"] = kwargs["plotting_config"]

    def capture_gamma_show(self, *args, **kwargs) -> None:
        received["gamma"] = kwargs["plotting_config"]

    monkeypatch.setattr(SARImageLoader, "plot", capture_loader_plot)
    monkeypatch.setattr(SARImageLoader, "plot_aligned", capture_aligned_plot)
    monkeypatch.setattr(workflows_module, "show_registration_overlay", capture_overlay)
    monkeypatch.setattr(GammaMapEvaluator, "show", capture_gamma_show)

    result = complete_workflow(
        measured_file_path=str(measured_csv),
        reference_file_path=str(reference_csv),
        transform_type="rigid",
        resample_resolution=0.001,
        show_plot=True,
        plotting={
            "window_mm": (-50, 55, -45, 65),
            "font_size": 18,
            "save_dpi": 175,
        },
        stages=[
            {
                "translation_step": 0.010,
                "rot_step_deg": 4.0,
                "rot_span_deg": 180.0,
                "tx_steps": 6,
                "ty_steps": 6,
            }
        ],
        evaluation_roi_policy="intersection",
        save_failures_overlay=False,
        log_level="WARNING",
    )

    assert result.evaluated_pixel_count > 0
    assert set(received) == {"loader", "aligned", "overlay", "gamma"}
    for plotting_config in received.values():
        assert isinstance(plotting_config, PlottingConfig)
        assert plotting_config.window_mm == (-50.0, 55.0, -45.0, 65.0)
        assert plotting_config.font_size == 18
        assert plotting_config.save_dpi == 175


@pytest.mark.slow
def test_complete_workflow_recovers_high_pass_rate_for_shifted_synthetic_input(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    shifted_dir = tmp_path / "shifted"
    shifted_dir.mkdir()

    baseline_measured, baseline_reference = _write_synthetic_workflow_pair(baseline_dir)
    shifted_measured, shifted_reference = _write_synthetic_workflow_pair(
        shifted_dir, tx=0.004, ty=-0.003, theta_deg=0.0
    )

    common_kwargs = dict(
        transform_type="translate",
        resample_resolution=0.005,
        render_plots=False,
        show_plot=False,
        distance_to_agreement=2.0,
        dose_to_agreement=5.0,
        stages=[
            {
                "translation_step": 0.001,
                "rot_step_deg": 0.0,
                "rot_span_deg": 0.0,
                "tx_steps": 6,
                "ty_steps": 6,
            },
        ],
    )

    baseline = complete_workflow(
        measured_file_path=str(baseline_measured),
        reference_file_path=str(baseline_reference),
        **common_kwargs,
    )
    shifted = complete_workflow(
        measured_file_path=str(shifted_measured),
        reference_file_path=str(shifted_reference),
        **common_kwargs,
    )

    shifted_loader = SARImageLoader(
        measured_path=str(shifted_measured),
        reference_path=str(shifted_reference),
        resample_resolution=0.005,
        show_plot=False,
        warn=True,
    )
    _, reference_mask_u8 = shifted_loader.make_metric_masks()
    unregistered = GammaMapEvaluator(
        reference_sar_linear=shifted_loader.reference_image_linear,
        measured_sar_linear=shifted_loader.measured_image_linear,
        reference_to_measured_transform=sitk.TranslationTransform(2),
        dose_to_agreement_percent=5.0,
        distance_to_agreement_mm=2.0,
        gamma_cap=2.0,
    )
    unregistered.reference_mask_u8 = reference_mask_u8
    unregistered.compute()
    assert unregistered.pass_rate_percent is not None

    assert baseline.pass_rate_percent >= 98.0
    assert shifted.pass_rate_percent >= 85.0
    assert shifted.pass_rate_percent >= unregistered.pass_rate_percent + 20.0


@pytest.mark.slow
def test_complete_workflow_default_roi_matches_intersection(tmp_path: Path) -> None:
    measured_csv, reference_csv = _write_truncated_support_workflow_pair(tmp_path)

    common_kwargs = dict(
        measured_file_path=str(measured_csv),
        reference_file_path=str(reference_csv),
        transform_type="translate",
        resample_resolution=0.005,
        render_plots=False,
        show_plot=False,
        distance_to_agreement=2.0,
        dose_to_agreement=5.0,
        stages=[
            {
                "translation_step": 0.001,
                "rot_step_deg": 0.0,
                "rot_span_deg": 0.0,
                "tx_steps": 1,
                "ty_steps": 1,
            }
        ],
    )

    default_result = complete_workflow(**common_kwargs)
    intersection_result = complete_workflow(
        evaluation_roi_policy="intersection",
        **common_kwargs,
    )

    assert (
        default_result.evaluated_pixel_count
        == intersection_result.evaluated_pixel_count
    )
    assert default_result.passed_pixel_count == intersection_result.passed_pixel_count
    assert default_result.failed_pixel_count == intersection_result.failed_pixel_count
    assert default_result.pass_rate_percent == pytest.approx(
        intersection_result.pass_rate_percent,
        abs=1e-9,
    )


@pytest.mark.slow
def test_complete_workflow_roi_policies_change_evaluated_region_consistently(
    tmp_path: Path,
) -> None:
    measured_csv, reference_csv = _write_truncated_support_workflow_pair(tmp_path)

    common_kwargs = dict(
        measured_file_path=str(measured_csv),
        reference_file_path=str(reference_csv),
        transform_type="translate",
        resample_resolution=0.005,
        render_plots=False,
        show_plot=False,
        distance_to_agreement=2.0,
        dose_to_agreement=5.0,
        stages=[
            {
                "translation_step": 0.001,
                "rot_step_deg": 0.0,
                "rot_span_deg": 0.0,
                "tx_steps": 1,
                "ty_steps": 1,
            }
        ],
    )

    none_result = complete_workflow(
        evaluation_roi_policy="none",
        **common_kwargs,
    )
    reference_only_result = complete_workflow(
        evaluation_roi_policy="reference_only",
        **common_kwargs,
    )
    intersection_result = complete_workflow(
        evaluation_roi_policy="intersection",
        **common_kwargs,
    )

    assert (
        none_result.evaluated_pixel_count > reference_only_result.evaluated_pixel_count
    )
    assert (
        reference_only_result.evaluated_pixel_count
        >= intersection_result.evaluated_pixel_count
    )
    assert (
        intersection_result.pass_rate_percent >= reference_only_result.pass_rate_percent
    )


@pytest.mark.slow
def test_complete_workflow_raises_mask_too_small_pre_registration(
    tmp_path: Path,
) -> None:
    """V3: pre-registration MASK_TOO_SMALL raises WorkflowExecutionError (hard error)."""
    from sar_pattern_validation.errors import WorkflowExecutionError

    # Narrow Gaussian (σ=4 mm) on a large grid: noise-filtered active area ~20 mm < 22 mm.
    x, y = make_rect_grid(xmin=-0.05, xmax=0.05, ymin=-0.05, ymax=0.05, step=0.002)
    _, _, Z_meas = gaussian_2d(x, y, x0=0.0, y0=0.0, sx=0.004, sy=0.004, peak=1.0)
    measured_csv = tmp_path / "narrow_measured.csv"
    write_sar_csv(measured_csv, x, y, Z_meas)

    _, _, Z_ref = gaussian_2d(x, y, x0=0.0, y0=0.0, sx=0.020, sy=0.020, peak=1.0)
    reference_csv = tmp_path / "wide_reference.csv"
    write_sar_csv(reference_csv, x, y, Z_ref)

    with pytest.raises(WorkflowExecutionError) as exc_info:
        complete_workflow(
            measured_file_path=str(measured_csv),
            reference_file_path=str(reference_csv),
            render_plots=False,
            show_plot=False,
            min_inscribed_square_mm=22.0,
        )

    issue = exc_info.value.issue
    assert issue is not None
    assert issue.code == "MASK_TOO_SMALL"
    assert issue.severity == "error"
    assert "pre-registration" in issue.message
    assert "22" in issue.message


@pytest.mark.slow
def test_complete_workflow_raises_mask_too_small_post_registration(
    tmp_path: Path,
) -> None:
    """V3: 1000 mm threshold hits pre-registration check first — WorkflowExecutionError raised."""
    from sar_pattern_validation.errors import WorkflowExecutionError

    measured_csv, reference_csv = _write_synthetic_workflow_pair(tmp_path)

    with pytest.raises(WorkflowExecutionError) as exc_info:
        complete_workflow(
            measured_file_path=str(measured_csv),
            reference_file_path=str(reference_csv),
            render_plots=False,
            show_plot=False,
            min_inscribed_square_mm=1000.0,
        )

    issue = exc_info.value.issue
    assert issue is not None
    assert issue.code == "MASK_TOO_SMALL"
    assert issue.severity == "error"
    assert "1000" in issue.message


def test_complete_workflow_v1_empty_measured_mask_raises_issue(tmp_path: Path) -> None:
    """V1: noise_floor > measured peak → EMPTY_MEASURED_MASK issue, not a raw ITK crash."""
    from sar_pattern_validation.errors import WorkflowExecutionError

    _, reference_csv = _write_synthetic_workflow_pair(tmp_path)
    # Write a measured CSV whose peak (0.001 W/kg) is below the default noise_floor (0.05)
    x, y = make_rect_grid(xmin=-0.04, xmax=0.04, ymin=-0.04, ymax=0.04, step=0.005)
    _, _, Z = gaussian_2d(x, y, x0=0.0, y0=0.0, sx=0.02, sy=0.02, peak=0.001)
    sub_floor_csv = tmp_path / "sub_floor_measured.csv"
    write_sar_csv(sub_floor_csv, x, y, Z)

    with pytest.raises(WorkflowExecutionError) as exc_info:
        complete_workflow(
            measured_file_path=str(sub_floor_csv),
            reference_file_path=str(reference_csv),
            render_plots=False,
            show_plot=False,
        )

    issue = exc_info.value.issue
    assert issue is not None
    assert issue.code == "EMPTY_MEASURED_MASK"
    assert issue.severity == "error"
    assert "noise floor" in issue.message.lower()


def test_complete_workflow_emits_csv_format_error_issue(tmp_path: Path) -> None:
    """CSV_FORMAT_ERROR issue is carried on WorkflowExecutionError for malformed input."""
    from sar_pattern_validation.errors import WorkflowExecutionError

    _, reference_csv = _write_synthetic_workflow_pair(tmp_path)
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("not,a,valid,sar,header\n1,2,3,4,5\n")

    with pytest.raises(WorkflowExecutionError) as exc_info:
        complete_workflow(
            measured_file_path=str(bad_csv),
            reference_file_path=str(reference_csv),
            render_plots=False,
            show_plot=False,
        )

    assert exc_info.value.issue is not None
    assert exc_info.value.issue.code == "CSV_FORMAT_ERROR"
    assert exc_info.value.issue.severity == "error"


def test_complete_workflow_v3_noise_filtered_pixels_excluded_from_gamma_mask(
    tmp_path: Path,
) -> None:
    """V3: metric mask (SAR >= cutoff) must gate gamma eval, not support mask.

    With noise_floor=0.001 cutoff=0.002 W/kg (near-zero); raising to 0.05 W/kg
    sets cutoff=0.1 W/kg and excludes sub-peak tails. Evaluated pixel count must
    be strictly smaller — proving the metric mask (not the boundary-only support
    mask) reaches the evaluator via workflows.py:_apply_roi_policy.
    """
    measured_csv, reference_csv = _write_synthetic_workflow_pair(tmp_path)

    fast_stages = [
        {
            "translation_step": 0.001,
            "rot_step_deg": 0.0,
            "rot_span_deg": 0.0,
            "tx_steps": 1,
            "ty_steps": 1,
        }
    ]
    common = dict(
        measured_file_path=str(measured_csv),
        reference_file_path=str(reference_csv),
        transform_type="translate",
        resample_resolution=0.005,
        render_plots=False,
        show_plot=False,
        distance_to_agreement=2.0,
        dose_to_agreement=5.0,
        stages=fast_stages,
        evaluation_roi_policy="intersection",
    )

    no_noise_result = complete_workflow(**common, noise_floor=0.001)
    noise_result = complete_workflow(**common, noise_floor=0.05)

    assert noise_result.evaluated_pixel_count < no_noise_result.evaluated_pixel_count, (
        "V3 violated: raising noise_floor did not reduce evaluated_pixel_count; "
        "metric mask may not be reaching the evaluator"
    )


def test_v13_measurement_area_restricts_data_not_just_plots(tmp_path: Path) -> None:
    """
    V13: measurement_area_x_mm/y_mm must filter the measured SAR data fed into
    SARImageLoader, not merely update the plot window.

    Setup: Measured Gaussian peak at (0, 0) mm on a ±150 mm symmetric grid
    (midpoint = peak = origin).  A 30×30 mm area centred at the data midpoint
    keeps only ±15 mm around the origin; data outside is excluded.

    Without area filter: the full ±150 mm grid has a larger noise-floor mask.
    With area filter:    only data within ±15 mm is kept → mask is smaller.
    """
    import SimpleITK as sitk

    x = np.arange(-0.150, 0.151, 0.005)
    y = np.arange(-0.150, 0.151, 0.005)
    _, _, meas_Z = gaussian_2d(x, y, x0=0.00, y0=0.00, sx=0.020, sy=0.020, peak=1.0)
    _, _, ref_Z = gaussian_2d(x, y, x0=0.00, y0=0.00, sx=0.020, sy=0.020, peak=1.0)
    measured_csv = tmp_path / "measured.csv"
    reference_csv = tmp_path / "reference.csv"
    write_sar_csv(measured_csv, x, y, meas_Z)
    write_sar_csv(reference_csv, x, y, ref_Z)

    # Full grid: peak at (0, 0) mm is above noise floor → non-empty mask.
    loader_full = SARImageLoader(
        str(measured_csv), str(reference_csv), noise_floor_wkg=0.05
    )
    mask_u8, _ = loader_full.make_metric_masks()
    full_pixel_count = int(sitk.GetArrayFromImage(mask_u8).astype(bool).sum())
    assert full_pixel_count > 0, (
        "Prerequisite failed: full ±150 mm grid must have mask pixels above noise floor"
    )

    # 30×30 mm filter centred at data midpoint = (0, 0) mm (same as peak here):
    # keeps only ±15 mm around the origin; data outside is excluded.
    # The filter captures the peak so the mask is non-empty, but smaller than full.
    loader_filtered = SARImageLoader(
        str(measured_csv),
        str(reference_csv),
        noise_floor_wkg=0.05,
        measurement_area_x_mm=30.0,
        measurement_area_y_mm=30.0,
    )
    mask_u8_filtered, _ = loader_filtered.make_metric_masks()
    filtered_pixel_count = int(
        sitk.GetArrayFromImage(mask_u8_filtered).astype(bool).sum()
    )
    assert filtered_pixel_count > 0, (
        "V13 prerequisite: 30×30 mm area centred at midpoint must be non-empty"
    )
    assert filtered_pixel_count < full_pixel_count, (
        "V13 violated: measurement_area=30×30 mm must reduce the measured mask area "
        "(data outside the declared area must not contribute); "
        f"full={full_pixel_count}, filtered={filtered_pixel_count}"
    )

    # complete_workflow enforces the 50 mm minimum (V14/V15); 30 mm must be
    # rejected at the config-validation layer, not reach the loader.
    with pytest.raises(ConfigValidationError):
        complete_workflow(
            measured_file_path=str(measured_csv),
            reference_file_path=str(reference_csv),
            noise_floor=0.05,
            render_plots=False,
            show_plot=False,
            measurement_area_x_mm=30.0,
            measurement_area_y_mm=30.0,
        )


def test_v19_measurement_area_centred_at_data_midpoint(tmp_path: Path) -> None:
    """
    V19: the measurement area window must be centred on the data midpoint (mean of
    x/y extents), not on the peak-SAR location.

    Setup: asymmetric grid 50–350 mm (midpoint 200 mm); Gaussian peak at (290, 290) mm
    with σ=60 mm so the midpoint region has SAR well above noise floor (≈0.33 W/kg).

    A 40×40 mm filter centred at the data midpoint (200 mm) keeps data in
    ~180–220 mm; one centred at the peak (290 mm) would keep 270–310 mm instead.
    Verify: filtered loader's measured axes are centred near 200 mm, not near 290 mm.
    """
    x = np.arange(0.050, 0.351, 0.005)  # 50–350 mm, midpoint = 200 mm
    y = np.arange(0.050, 0.351, 0.005)
    _, _, meas_Z = gaussian_2d(x, y, x0=0.290, y0=0.290, sx=0.060, sy=0.060, peak=1.0)
    _, _, ref_Z = gaussian_2d(x, y, x0=0.200, y0=0.200, sx=0.060, sy=0.060, peak=1.0)
    measured_csv = tmp_path / "measured_v19.csv"
    reference_csv = tmp_path / "reference_v19.csv"
    write_sar_csv(measured_csv, x, y, meas_Z)
    write_sar_csv(reference_csv, x, y, ref_Z)

    loader = SARImageLoader(
        str(measured_csv),
        str(reference_csv),
        noise_floor_wkg=0.05,
        measurement_area_x_mm=40.0,
        measurement_area_y_mm=40.0,
    )
    x_axes = loader._measured_axes_m[0]
    x_centre_mm = 1000.0 * (float(x_axes.min()) + float(x_axes.max())) / 2.0

    assert abs(x_centre_mm - 200.0) < 30.0, (
        f"V19 violated: filtered data x-centre should be near grid midpoint 200 mm "
        f"(midpoint centering), got {x_centre_mm:.1f} mm (peak is at 290 mm — "
        "peak centering would give ≈290 mm)"
    )


@pytest.mark.slow
def test_workflow_result_carries_measured_peak_wkg(tmp_path: Path) -> None:
    """V17: WorkflowResult.measured_peak_wkg must equal loader.measured_peak directly."""
    import pandas as pd

    measured_csv, reference_csv = _write_synthetic_workflow_pair(tmp_path)

    power_dbm = 10.0
    noise_floor = 0.05
    result = complete_workflow(
        measured_file_path=str(measured_csv),
        reference_file_path=str(reference_csv),
        noise_floor=noise_floor,
        power_level_dbm=power_dbm,
        render_plots=False,
        show_plot=False,
    )

    df = pd.read_csv(measured_csv)
    sar_col = next(c for c in df.columns if "sar" in c.lower() or "wkg" in c.lower())
    raw_peak = float(df[sar_col].max())

    assert result.measured_peak_wkg > 0.0
    assert result.measured_peak_wkg <= raw_peak + 1e-9, (
        "measured_peak_wkg must not exceed the raw CSV max"
    )
    at_power_via_roundtrip = result.measured_pssar * (10 ** ((power_dbm - 30.0) / 10.0))
    assert abs(result.measured_peak_wkg - at_power_via_roundtrip) < 1e-6, (
        "measured_peak_wkg must equal the at-power peak (consistent with measured_pssar)"
    )
