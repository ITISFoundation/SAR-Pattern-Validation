from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal

from sar_pattern_validation.registration2d import Transform2D

DEFAULT_MEASURED_FILE_PATH: Final[str] = "measured.csv"
DEFAULT_REFERENCE_FILE_PATH: Final[str] = "reference.csv"
DEFAULT_POWER_LEVEL_DBM: Final[float] = 30.0
DEFAULT_NOISE_FLOOR: Final[float] = 0.05
NOISE_FLOOR_MAX: Final[float] = 0.1
DEFAULT_SHOW_PLOT: Final[bool] = False
DEFAULT_RENDER_PLOTS: Final[bool] = True
DEFAULT_DOSE_TO_AGREEMENT: Final[float] = 5.0
DEFAULT_DISTANCE_TO_AGREEMENT: Final[float] = 2.0
DEFAULT_GAMMA_CAP: Final[float] = 2.0
DEFAULT_EVALUATION_ROI_POLICY: Final[
    Literal["reference_only", "intersection", "none"]
] = "intersection"
DEFAULT_REGISTRATION_STAGE_POLICY: Final[Literal["static", "adaptive"]] = "adaptive"
DEFAULT_ADAPTIVE_ASSUME_AXIAL_SYMMETRY: Final[bool] = True
DEFAULT_ADAPTIVE_MAX_STAGES: Final[int] = 5
DEFAULT_ADAPTIVE_MAX_STAGE_EVALS: Final[int] = 50000
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

# Minimum axis-aligned physical square (mm) that must fit within the gamma
# evaluation mask for the comparison to be considered valid. 22 mm = the
# face of the 10 g averaging cube. Per MGD 2026-04-24 feedback, slide 7.
DEFAULT_MIN_INSCRIBED_SQUARE_MM: Final[float] = 22.0
DEFAULT_PLOT_WINDOW_MM: Final[tuple[float, float, float, float]] = (
    -120.0,
    120.0,
    -120.0,
    120.0,
)

# Minimum measurement area: 50 mm on each axis (V14/V15 §B13).
# 0 is reserved for "auto" (no crop); any non-zero value must be ≥ 50 mm.
MEASUREMENT_AREA_MIN_MM_EXCLUSIVE: Final[float] = 50.0
MEASUREMENT_AREA_MAX_X_MM: Final[float] = 600.0
MEASUREMENT_AREA_MAX_Y_MM: Final[float] = 400.0
DEFAULT_PLOT_FONT_SIZE: Final[float] = 14.0
DEFAULT_SINGLE_FIGURE_SIZE: Final[tuple[float, float]] = (6.0, 6.0)
DEFAULT_COMBINED_FIGURE_SIZE: Final[tuple[float, float]] = (12.0, 5.0)
DEFAULT_PLOT_FIGURE_FACECOLOR: Final[str] = "white"
DEFAULT_PLOT_DARK_AXES_FACECOLOR: Final[str] = "black"
DEFAULT_PLOT_LIGHT_AXES_FACECOLOR: Final[str] = "white"
DEFAULT_PLOT_SAVE_DPI: Final[int] = 200
DEFAULT_PLOT_NOT_EVALUATED_COLOR: Final[str] = "lightgray"
DEFAULT_PLOT_CROPPED_DATA_COLOR: Final[str] = "#555555"
DEFAULT_PLOT_NOISE_FLOOR_COLOR: Final[str] = "#9aa0a6"

ROI_POLICY_CHOICES: Final[tuple[str, ...]] = ("reference_only", "intersection", "none")
EvaluationRoiPolicy = Literal["reference_only", "intersection", "none"]
LOG_LEVEL_CHOICES: Final[tuple[str, ...]] = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
)


def default_registration_stages() -> list[dict[str, Any]]:
    return [
        dict(
            translation_step=0.010,
            rot_step_deg=4.0,
            rot_span_deg=180.0,
            tx_steps=6,
            ty_steps=6,
        ),
        dict(
            translation_step=0.001,
            rot_step_deg=1.0,
            rot_span_deg=90.0,
            tx_steps=6,
            ty_steps=6,
        ),
    ]


@dataclass
class PlottingConfig:
    window_mm: tuple[float, float, float, float] = DEFAULT_PLOT_WINDOW_MM
    font_size: float = DEFAULT_PLOT_FONT_SIZE
    single_figure_size: tuple[float, float] = DEFAULT_SINGLE_FIGURE_SIZE
    combined_figure_size: tuple[float, float] = DEFAULT_COMBINED_FIGURE_SIZE
    figure_facecolor: str = DEFAULT_PLOT_FIGURE_FACECOLOR
    dark_axes_facecolor: str = DEFAULT_PLOT_DARK_AXES_FACECOLOR
    light_axes_facecolor: str = DEFAULT_PLOT_LIGHT_AXES_FACECOLOR
    save_dpi: int = DEFAULT_PLOT_SAVE_DPI
    not_evaluated_color: str = DEFAULT_PLOT_NOT_EVALUATED_COLOR
    cropped_data_color: str = DEFAULT_PLOT_CROPPED_DATA_COLOR
    noise_floor_color: str = DEFAULT_PLOT_NOISE_FLOOR_COLOR
    measurement_area_x_mm: float | None = None
    measurement_area_y_mm: float | None = None
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0
    save_colorbars: bool = True


@dataclass
class WorkflowConfig:
    measured_file_path: str = DEFAULT_MEASURED_FILE_PATH
    reference_file_path: str = DEFAULT_REFERENCE_FILE_PATH
    power_level_dbm: float = DEFAULT_POWER_LEVEL_DBM

    loaded_images_save_path: str | None = None
    reference_image_save_path: str | None = None
    measured_image_save_path: str | None = None
    aligned_meas_save_path: str | None = None
    registered_image_save_path: str | None = None
    gamma_comparison_image_path: str | None = None

    resample_resolution: float | None = None
    noise_floor: float = DEFAULT_NOISE_FLOOR
    show_plot: bool = DEFAULT_SHOW_PLOT
    render_plots: bool = DEFAULT_RENDER_PLOTS

    transform_type: Transform2D = Transform2D.RIGID
    stages: list[dict[str, Any]] = field(default_factory=default_registration_stages)
    registration_stage_policy: Literal["static", "adaptive"] = (
        DEFAULT_REGISTRATION_STAGE_POLICY
    )
    adaptive_assume_axial_symmetry: bool = DEFAULT_ADAPTIVE_ASSUME_AXIAL_SYMMETRY
    adaptive_max_stages: int = DEFAULT_ADAPTIVE_MAX_STAGES
    adaptive_max_stage_evals: int = DEFAULT_ADAPTIVE_MAX_STAGE_EVALS

    dose_to_agreement: float = DEFAULT_DOSE_TO_AGREEMENT
    distance_to_agreement: float = DEFAULT_DISTANCE_TO_AGREEMENT
    gamma_cap: float = DEFAULT_GAMMA_CAP
    evaluation_roi_policy: EvaluationRoiPolicy = DEFAULT_EVALUATION_ROI_POLICY

    save_failures_overlay: bool = True
    log_level: str = DEFAULT_LOG_LEVEL
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    output_dir: str | None = None
    measurement_area_x_mm: float | None = None
    measurement_area_y_mm: float | None = None
    min_inscribed_square_mm: float = DEFAULT_MIN_INSCRIBED_SQUARE_MM
