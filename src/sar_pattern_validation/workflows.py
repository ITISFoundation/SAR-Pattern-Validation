from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from sar_pattern_validation.errors import (
    CsvFormatError,
    ValidationIssue,
    WorkflowExecutionError,
)
from sar_pattern_validation.gamma_eval import (
    GammaMapEvaluator,
    _mask_fits_axis_aligned_square_mm,
)
from sar_pattern_validation.image_loader import SARImageLoader
from sar_pattern_validation.plotting import show_registration_overlay
from sar_pattern_validation.registration2d import Rigid2DRegistration, Transform2D
from sar_pattern_validation.utils import configure_root_logging, ensure_output_path
from sar_pattern_validation.workflow_config import (
    DEFAULT_ADAPTIVE_ASSUME_AXIAL_SYMMETRY,
    DEFAULT_ADAPTIVE_MAX_STAGE_EVALS,
    DEFAULT_ADAPTIVE_MAX_STAGES,
    DEFAULT_COMBINED_FIGURE_SIZE,
    DEFAULT_DISTANCE_TO_AGREEMENT,
    DEFAULT_DOSE_TO_AGREEMENT,
    DEFAULT_GAMMA_CAP,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MEASURED_FILE_PATH,
    DEFAULT_NOISE_FLOOR,
    DEFAULT_PLOT_DARK_AXES_FACECOLOR,
    DEFAULT_PLOT_FIGURE_FACECOLOR,
    DEFAULT_PLOT_FONT_SIZE,
    DEFAULT_PLOT_LIGHT_AXES_FACECOLOR,
    DEFAULT_PLOT_SAVE_DPI,
    DEFAULT_PLOT_WINDOW_MM,
    DEFAULT_POWER_LEVEL_DBM,
    DEFAULT_REFERENCE_FILE_PATH,
    DEFAULT_REGISTRATION_STAGE_POLICY,
    DEFAULT_RENDER_PLOTS,
    DEFAULT_SINGLE_FIGURE_SIZE,
    ROI_POLICY_CHOICES,
    WorkflowConfig,
)
from sar_pattern_validation.workflow_schema import (
    ensure_input_files_exist,
    validate_workflow_config,
)

LOGGER = logging.getLogger(__name__)


class WorkflowResultCLIExcludedFields(Enum):
    """
    Fields to exclude from CLI JSON serialization.

    Note: Update these values if WorkflowResult field names change.
    """

    GAMMA_MAP = "gamma_map"
    EVALUATION_MASK = "evaluation_mask"


@dataclass
class WorkflowResult:
    pass_rate_percent: float
    evaluated_pixel_count: int
    passed_pixel_count: int
    failed_pixel_count: int
    gamma_image_path: Path | None
    failure_image_path: Path | None
    registered_overlay_path: Path | None
    loaded_images_path: Path | None
    reference_image_path: Path | None
    measured_image_path: Path | None
    aligned_measured_path: Path | None
    measured_peak_wkg: float
    measured_pssar: float
    reference_pssar: float
    scaling_error: float
    dose_to_agreement: float
    distance_to_agreement: float
    min_inscribed_square_mm: float
    mask_fits_min_inscribed_square: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    gamma_map: np.ndarray | None = field(default=None, compare=False)
    evaluation_mask: np.ndarray | None = field(default=None, compare=False)


def _write_output_dir(
    output_dir: Path,
    result: WorkflowResult,
    evaluator: GammaMapEvaluator,
) -> None:
    """
    Write artifacts to output_dir:
      - results.json    (scalar WorkflowResult fields)
      - gamma_map.npy   (gamma map array)
      - gamma_map.png   (gamma map visualisation)
      - failure_map.png (failure map visualisation)
    """
    import dataclasses

    output_dir.mkdir(parents=True, exist_ok=True)

    excluded = {f.value for f in WorkflowResultCLIExcludedFields}
    scalar_fields: dict[str, Any] = {}
    for f in dataclasses.fields(result):
        if f.name in excluded:
            continue
        v = getattr(result, f.name)
        scalar_fields[f.name] = str(v) if isinstance(v, Path) else v

    (output_dir / "results.json").write_text(
        json.dumps(scalar_fields, indent=2), encoding="utf-8"
    )

    if result.gamma_map is not None:
        np.save(output_dir / "gamma_map.npy", result.gamma_map)

    evaluator.show(
        gamma_image_save_path=output_dir / "gamma_map.png",
        failure_image_save_path=output_dir / "failure_map.png",
    )


def _failure_overlay_path(
    gamma_comparison_image_path: Path | None, save_failures_overlay: bool
) -> Path | None:
    if not save_failures_overlay:
        return None
    if gamma_comparison_image_path is None:
        LOGGER.warning(
            "save_failures_overlay is enabled but gamma_comparison_image_path is not set. "
            "Failure overlay will not be saved."
        )
        return None
    return gamma_comparison_image_path.with_name(
        f"{gamma_comparison_image_path.stem}_failures{gamma_comparison_image_path.suffix}"
    )


def _apply_roi_policy(
    evaluator: GammaMapEvaluator,
    *,
    reference_mask_u8: sitk.Image,
    measured_mask_u8: sitk.Image,
    policy: str,
) -> None:
    if policy == "reference_only":
        evaluator.reference_mask_u8 = reference_mask_u8
        evaluator.measured_mask_u8 = None
        return
    if policy == "intersection":
        evaluator.reference_mask_u8 = reference_mask_u8
        evaluator.measured_mask_u8 = measured_mask_u8
        return
    if policy == "none":
        evaluator.reference_mask_u8 = None
        evaluator.measured_mask_u8 = None
        return
    raise ValueError(f"Unsupported evaluation_roi_policy: {policy}")


def _complete_workflow(config: WorkflowConfig) -> WorkflowResult:
    try:
        ensure_input_files_exist(config)

        loaded_images_save_path = (
            ensure_output_path(config.loaded_images_save_path)
            if config.render_plots
            else None
        )
        reference_image_save_path = (
            ensure_output_path(config.reference_image_save_path)
            if config.render_plots
            else None
        )
        measured_image_save_path = (
            ensure_output_path(config.measured_image_save_path)
            if config.render_plots
            else None
        )
        aligned_meas_save_path = (
            ensure_output_path(config.aligned_meas_save_path)
            if config.render_plots
            else None
        )
        registered_image_save_path = (
            ensure_output_path(config.registered_image_save_path)
            if config.render_plots
            else None
        )
        gamma_comparison_image_path = (
            ensure_output_path(config.gamma_comparison_image_path)
            if config.render_plots
            else None
        )
        failure_image_path = _failure_overlay_path(
            gamma_comparison_image_path, config.save_failures_overlay
        )

        LOGGER.info("Step 1/3: Loading and preprocessing SAR images")
        loader = SARImageLoader(
            measured_path=config.measured_file_path,
            reference_path=config.reference_file_path,
            power_level_dbm=config.power_level_dbm,
            resample_resolution=config.resample_resolution,
            noise_floor_wkg=config.noise_floor,
            show_plot=False,
            warn=True,
            measurement_area_x_mm=config.measurement_area_x_mm,
            measurement_area_y_mm=config.measurement_area_y_mm,
        )

        reference_db, measured_db = loader.get_images()

        # Auto-center plot window on the measured data centroid only when the
        # caller left window_mm at its default value (no explicit override).
        _cx_mm = float(loader._measured_axes_m[0].mean()) * 1000.0
        _cy_mm = float(loader._measured_axes_m[1].mean()) * 1000.0
        config.plotting.center_x_mm = _cx_mm
        config.plotting.center_y_mm = _cy_mm
        if config.plotting.window_mm == DEFAULT_PLOT_WINDOW_MM:
            _x_span_mm = (
                loader._measured_axes_m[0].max() - loader._measured_axes_m[0].min()
            ) * 1000.0
            _y_span_mm = (
                loader._measured_axes_m[1].max() - loader._measured_axes_m[1].min()
            ) * 1000.0
            if (
                config.plotting.measurement_area_x_mm is not None
                and config.plotting.measurement_area_y_mm is not None
            ):
                _half = (
                    max(
                        config.plotting.measurement_area_x_mm,
                        config.plotting.measurement_area_y_mm,
                    )
                    / 2.0
                )
            else:
                _half = max(_x_span_mm, _y_span_mm) / 2.0 * 1.1
            config.plotting.window_mm = (
                _cx_mm - _half,
                _cx_mm + _half,
                _cy_mm - _half,
                _cy_mm + _half,
            )

        if config.render_plots and (
            config.show_plot
            or loaded_images_save_path is not None
            or reference_image_save_path is not None
            or measured_image_save_path is not None
        ):
            loader.plot(
                image_save_path=loaded_images_save_path,
                reference_save_path=reference_image_save_path,
                measured_save_path=measured_image_save_path,
                plotting_config=config.plotting,
            )

        LOGGER.info("Step 2/3: Registering reference SAR onto measured grid")
        measured_mask_u8, reference_mask_u8 = loader.make_metric_masks()
        measured_support_u8, _ = loader.make_support_masks()

        if not np.any(sitk.GetArrayFromImage(measured_mask_u8)):
            _issue = ValidationIssue(
                severity="error",
                code="EMPTY_MEASURED_MASK",
                message=(
                    f"No measured SAR values exceed the noise floor "
                    f"({config.noise_floor:.3f} W/kg; measured peak: "
                    f"{loader.measured_raw_peak:.4g} W/kg). "
                    f"Lower the noise floor or check the measurement file."
                ),
            )
            raise WorkflowExecutionError(_issue.message, issue=_issue)

        # V3: pre-registration check — noise-filtered measured mask must admit a
        # min_inscribed_square_mm × min_inscribed_square_mm axis-aligned square.
        meas_arr = sitk.GetArrayFromImage(measured_mask_u8).astype(bool)
        if not _mask_fits_axis_aligned_square_mm(
            mask=meas_arr,
            side_mm=config.min_inscribed_square_mm,
            spacing_m=measured_mask_u8.GetSpacing(),
        ):
            _issue = ValidationIssue(
                severity="error",
                code="MASK_TOO_SMALL",
                message=(
                    f"Noise-filtered measured mask (pre-registration) does not contain a "
                    f"{config.min_inscribed_square_mm:.0f} mm × "
                    f"{config.min_inscribed_square_mm:.0f} mm axis-aligned inscribed "
                    f"square. The gamma comparison is invalid. "
                    f"If the mask is small due to noise filtering, try lowering the noise floor threshold."
                ),
            )
            raise WorkflowExecutionError(_issue.message, issue=_issue)

        reg = Rigid2DRegistration(
            fixed_image=measured_db,
            moving_image=reference_db,
            transform_type=config.transform_type,
        )

        registration_stages = config.stages
        if config.registration_stage_policy == "adaptive":
            registration_stages = reg.build_adaptive_stages(
                fixed_image=measured_db,
                moving_image=reference_db,
                transform_type=config.transform_type,
                fixed_mask=measured_mask_u8,
                moving_mask=reference_mask_u8,
                assume_axial_symmetry=config.adaptive_assume_axial_symmetry,
                max_stages=config.adaptive_max_stages,
                max_stage_evals=config.adaptive_max_stage_evals,
            )
            LOGGER.info("Adaptive registration stages: %s", registration_stages)

        aligned_db, final_tx = reg.run(
            stages=registration_stages,
            fixed_mask=measured_mask_u8,
            moving_mask=reference_mask_u8,
        )

        if config.render_plots and (
            config.show_plot or aligned_meas_save_path is not None
        ):
            aligned_lin = reg.db_to_linear(aligned_db, floor_norm=-120.0)
            loader.plot_aligned(
                aligned_lin,
                aligned_meas_save_path,
                plotting_config=config.plotting,
            )

        if config.render_plots:
            show_registration_overlay(
                measured_db,
                aligned_db,
                title="Rigid Registration Overlay",
                image_save_path=registered_image_save_path,
                noise_floor_mask=loader._measured_noise_floor_mask,
                plotting_config=config.plotting,
            )

        LOGGER.info(
            "Step 3/3: Computing gamma map (policy=%s, dta=%.3f mm, dose=%.3f%%)",
            config.evaluation_roi_policy,
            config.distance_to_agreement,
            config.dose_to_agreement,
        )
        evaluator = GammaMapEvaluator(
            reference_sar_linear=loader.reference_image_linear,
            measured_sar_linear=loader.measured_image_linear,
            reference_to_measured_transform=final_tx,
            dose_to_agreement_percent=float(config.dose_to_agreement),
            distance_to_agreement_mm=float(config.distance_to_agreement),
            gamma_cap=float(config.gamma_cap),
        )
        _apply_roi_policy(
            evaluator,
            reference_mask_u8=reference_mask_u8,
            measured_mask_u8=measured_mask_u8,
            policy=config.evaluation_roi_policy,
        )

        evaluator.compute()
        gamma_map = evaluator.gamma_map
        evaluation_mask = evaluator.evaluation_mask
        if config.render_plots:
            evaluator.show(
                gamma_image_save_path=gamma_comparison_image_path,
                failure_image_save_path=failure_image_path,
                noise_floor_mask=loader._measured_noise_floor_mask,
                plotting_config=config.plotting,
            )

        if (
            evaluator.pass_rate_percent is None
            or evaluator.evaluated_pixel_count is None
            or evaluator.passed_pixel_count is None
            or evaluator.failed_pixel_count is None
        ):
            raise RuntimeError("Gamma evaluation completed without summary metrics.")

        # Per MGD 2026-04-24 feedback (slide 7): the gamma comparison is only
        # valid when an axis-aligned square of `min_inscribed_square_mm` fits
        # entirely inside the (post-registration, post-noise-filter) mask.
        mask_fits_min_inscribed_square = (
            evaluator.evaluation_mask_fits_axis_aligned_square_mm(
                config.min_inscribed_square_mm
            )
        )
        if not mask_fits_min_inscribed_square:
            _issue = ValidationIssue(
                severity="error",
                code="MASK_TOO_SMALL",
                message=(
                    f"Gamma evaluation mask does not contain a "
                    f"{config.min_inscribed_square_mm:.0f} mm × "
                    f"{config.min_inscribed_square_mm:.0f} mm axis-aligned inscribed "
                    f"square. The gamma comparison is invalid. "
                    f"If the mask is small due to noise filtering, try lowering the noise floor threshold."
                ),
            )
            raise WorkflowExecutionError(_issue.message, issue=_issue)

        LOGGER.info(
            "Gamma completed: pass_rate=%.2f%%, evaluated=%d, passed=%d, failed=%d, "
            "mask_fits_min_inscribed_square=%s",
            evaluator.pass_rate_percent,
            evaluator.evaluated_pixel_count,
            evaluator.passed_pixel_count,
            evaluator.failed_pixel_count,
            mask_fits_min_inscribed_square,
        )
        workflow_result = WorkflowResult(
            pass_rate_percent=evaluator.pass_rate_percent,
            evaluated_pixel_count=evaluator.evaluated_pixel_count,
            passed_pixel_count=evaluator.passed_pixel_count,
            failed_pixel_count=evaluator.failed_pixel_count,
            gamma_image_path=gamma_comparison_image_path,
            failure_image_path=failure_image_path,
            registered_overlay_path=registered_image_save_path,
            loaded_images_path=loaded_images_save_path,
            reference_image_path=reference_image_save_path,
            measured_image_path=measured_image_save_path,
            aligned_measured_path=aligned_meas_save_path,
            measured_peak_wkg=loader.measured_peak,
            measured_pssar=loader.measured_peak_30dbm,
            reference_pssar=loader.reference_peak,
            scaling_error=loader.scaling_error,
            dose_to_agreement=config.dose_to_agreement,
            distance_to_agreement=config.distance_to_agreement,
            min_inscribed_square_mm=config.min_inscribed_square_mm,
            mask_fits_min_inscribed_square=mask_fits_min_inscribed_square,
            gamma_map=gamma_map,
            evaluation_mask=evaluation_mask,
        )

        if config.output_dir is not None:
            _write_output_dir(Path(config.output_dir), workflow_result, evaluator)

        return workflow_result
    except WorkflowExecutionError:
        raise
    except CsvFormatError as exc:
        _issue = ValidationIssue(
            severity="error",
            code="CSV_FORMAT_ERROR",
            message=str(exc),
        )
        raise WorkflowExecutionError(f"Workflow failed: {exc}", issue=_issue) from exc
    except Exception as exc:
        raise WorkflowExecutionError(f"Workflow failed: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    defaults = WorkflowConfig()
    parser = argparse.ArgumentParser(description="Run a SAR gamma comparison workflow.")
    parser.add_argument(
        "--measured_file_path",
        type=str,
        default=DEFAULT_MEASURED_FILE_PATH,
    )
    parser.add_argument(
        "--reference_file_path",
        dest="reference_file_path",
        type=str,
        default=DEFAULT_REFERENCE_FILE_PATH,
    )
    parser.add_argument(
        "--power_level_dbm",
        type=float,
        default=DEFAULT_POWER_LEVEL_DBM,
    )
    parser.add_argument("--loaded_images_save_path", type=str, default=None)
    parser.add_argument("--reference_image_save_path", type=str, default=None)
    parser.add_argument("--measured_image_save_path", type=str, default=None)
    parser.add_argument("--aligned_meas_save_path", type=str, default=None)
    parser.add_argument("--registered_image_save_path", type=str, default=None)
    parser.add_argument("--gamma_comparison_image_path", type=str, default=None)

    parser.add_argument("--resample_resolution", type=float, default=None)
    parser.add_argument("--noise_floor", type=float, default=DEFAULT_NOISE_FLOOR)
    parser.add_argument(
        "--show_plot",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_plot,
    )
    parser.add_argument(
        "--render_plots",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RENDER_PLOTS,
    )

    parser.add_argument(
        "--transform_type",
        type=Transform2D,
        choices=list(Transform2D),
        default=defaults.transform_type,
    )
    parser.add_argument(
        "--registration_stage_policy",
        type=str,
        choices=["static", "adaptive"],
        default=DEFAULT_REGISTRATION_STAGE_POLICY,
    )
    parser.add_argument(
        "--adaptive_assume_axial_symmetry",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ADAPTIVE_ASSUME_AXIAL_SYMMETRY,
    )
    parser.add_argument(
        "--adaptive_max_stages",
        type=int,
        default=DEFAULT_ADAPTIVE_MAX_STAGES,
    )
    parser.add_argument(
        "--adaptive_max_stage_evals",
        type=int,
        default=DEFAULT_ADAPTIVE_MAX_STAGE_EVALS,
    )

    parser.add_argument(
        "--dose_to_agreement",
        type=float,
        default=DEFAULT_DOSE_TO_AGREEMENT,
    )
    parser.add_argument(
        "--distance_to_agreement",
        type=float,
        default=DEFAULT_DISTANCE_TO_AGREEMENT,
    )
    parser.add_argument("--gamma_cap", type=float, default=DEFAULT_GAMMA_CAP)
    parser.add_argument(
        "--evaluation_roi_policy",
        type=str,
        default=defaults.evaluation_roi_policy,
        choices=list(ROI_POLICY_CHOICES),
    )
    parser.add_argument(
        "--save_failures_overlay",
        action=argparse.BooleanOptionalAction,
        default=defaults.save_failures_overlay,
    )
    parser.add_argument(
        "--plot-window-mm",
        nargs=4,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    parser.add_argument("--plot-font-size", type=float, default=None)
    parser.add_argument(
        "--plot-single-figure-size",
        nargs=2,
        type=float,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--plot-combined-figure-size",
        nargs=2,
        type=float,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--plot-figure-facecolor", type=str, default=None)
    parser.add_argument("--plot-dark-axes-facecolor", type=str, default=None)
    parser.add_argument("--plot-light-axes-facecolor", type=str, default=None)
    parser.add_argument("--plot-save-dpi", type=int, default=None)
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate the LaTeX validation report into <report_output_dir>/.",
    )
    parser.add_argument(
        "--report_output_dir",
        type=str,
        default=None,
        help="Output directory for the generated LaTeX report (default: ./report).",
    )
    parser.add_argument(
        "--report_template_dir",
        type=str,
        default=None,
        help="Override path to the LaTeX report template directory.",
    )
    parser.add_argument(
        "--report_antenna_type",
        type=str,
        default="dipole",
    )
    parser.add_argument("--report_frequency_mhz", type=int, default=0)
    parser.add_argument("--report_distance_mm", type=int, default=0)
    parser.add_argument("--report_mass_g", type=int, default=0)
    parser.add_argument(
        "--measurement_area_x_mm",
        type=float,
        default=None,
        help="Measurement area x dimension (mm). Must be > 22 and <= 600.",
    )
    parser.add_argument(
        "--measurement_area_y_mm",
        type=float,
        default=None,
        help="Measurement area y dimension (mm). Must be > 22 and <= 400.",
    )
    parser.add_argument(
        "--min_inscribed_square_mm",
        type=float,
        default=defaults.min_inscribed_square_mm,
        help=(
            "Minimum axis-aligned square (mm) that must fit within the gamma "
            "evaluation mask for the comparison to be valid."
        ),
    )
    parser.add_argument("--log_level", type=str, default=DEFAULT_LOG_LEVEL)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="If set, write results.json, gamma_map.npy, gamma_map.png, and failure_map.png here.",
    )
    return parser


def _normalize_plotting_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw_config)

    plotting = normalized.get("plotting")
    if plotting is None:
        plotting = {}
    elif is_dataclass(plotting):
        plotting = asdict(plotting)
    elif hasattr(plotting, "items"):
        plotting = dict(plotting.items())
    else:
        plotting = dict(plotting)

    cli_plotting = {
        "window_mm": normalized.pop("plot_window_mm", None),
        "font_size": normalized.pop("plot_font_size", None),
        "single_figure_size": normalized.pop("plot_single_figure_size", None),
        "combined_figure_size": normalized.pop("plot_combined_figure_size", None),
        "figure_facecolor": normalized.pop("plot_figure_facecolor", None),
        "dark_axes_facecolor": normalized.pop("plot_dark_axes_facecolor", None),
        "light_axes_facecolor": normalized.pop("plot_light_axes_facecolor", None),
        "save_dpi": normalized.pop("plot_save_dpi", None),
    }

    defaults = {
        "window_mm": DEFAULT_PLOT_WINDOW_MM,
        "font_size": DEFAULT_PLOT_FONT_SIZE,
        "single_figure_size": DEFAULT_SINGLE_FIGURE_SIZE,
        "combined_figure_size": DEFAULT_COMBINED_FIGURE_SIZE,
        "figure_facecolor": DEFAULT_PLOT_FIGURE_FACECOLOR,
        "dark_axes_facecolor": DEFAULT_PLOT_DARK_AXES_FACECOLOR,
        "light_axes_facecolor": DEFAULT_PLOT_LIGHT_AXES_FACECOLOR,
        "save_dpi": DEFAULT_PLOT_SAVE_DPI,
    }
    for key, value in cli_plotting.items():
        if value is not None:
            plotting[key] = tuple(value) if isinstance(value, list) else value

    normalized["plotting"] = {**defaults, **plotting}
    return normalized


_REPORT_KEYS = (
    "report",
    "report_output_dir",
    "report_template_dir",
    "report_antenna_type",
    "report_frequency_mhz",
    "report_distance_mm",
    "report_mass_g",
)


def complete_workflow(*args, **kwargs) -> WorkflowResult:
    parser = _build_parser()

    raw_config: dict[str, Any]
    if kwargs:
        raw_config = kwargs
    else:
        parsed = parser.parse_args([str(a) for a in args] if args else None)
        raw_config = vars(parsed)

    report_settings = {key: raw_config.pop(key, None) for key in _REPORT_KEYS}

    raw_config = _normalize_plotting_config(raw_config)

    config = validate_workflow_config(raw_config)

    configure_root_logging(config.log_level)

    LOGGER.info("WORKFLOW START with config: %s", config)
    result = _complete_workflow(config)

    if report_settings.get("report"):
        from sar_pattern_validation.report import (
            DEFAULT_TEMPLATE_DIR,
            generate_report,
        )

        report_path = generate_report(
            workflow_result=result,
            workflow_config=config,
            output_dir=report_settings.get("report_output_dir") or "report",
            template_dir=report_settings.get("report_template_dir")
            or DEFAULT_TEMPLATE_DIR,
            antenna_type=report_settings.get("report_antenna_type") or "dipole",
            frequency_mhz=report_settings.get("report_frequency_mhz") or 0,
            distance_mm=report_settings.get("report_distance_mm") or 0,
            mass_g=report_settings.get("report_mass_g") or 0,
        )
        LOGGER.info("Report written to: %s", report_path)

    return result


if __name__ == "__main__":
    complete_workflow()
