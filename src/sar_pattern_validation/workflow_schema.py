from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from sar_pattern_validation.errors import ConfigValidationError
from sar_pattern_validation.registration2d import Transform2D
from sar_pattern_validation.workflow_config import (
    DEFAULT_ADAPTIVE_ASSUME_AXIAL_SYMMETRY,
    DEFAULT_ADAPTIVE_MAX_STAGE_EVALS,
    DEFAULT_ADAPTIVE_MAX_STAGES,
    DEFAULT_COMBINED_FIGURE_SIZE,
    DEFAULT_DISTANCE_TO_AGREEMENT,
    DEFAULT_DOSE_TO_AGREEMENT,
    DEFAULT_EVALUATION_ROI_POLICY,
    DEFAULT_GAMMA_CAP,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MEASURED_FILE_PATH,
    DEFAULT_MIN_INSCRIBED_SQUARE_MM,
    DEFAULT_NOISE_FLOOR,
    DEFAULT_PLOT_DARK_AXES_FACECOLOR,
    DEFAULT_PLOT_FIGURE_FACECOLOR,
    DEFAULT_PLOT_FONT_SIZE,
    DEFAULT_PLOT_LIGHT_AXES_FACECOLOR,
    DEFAULT_PLOT_NOT_EVALUATED_COLOR,
    DEFAULT_PLOT_SAVE_DPI,
    DEFAULT_PLOT_WINDOW_MM,
    DEFAULT_POWER_LEVEL_DBM,
    DEFAULT_REFERENCE_FILE_PATH,
    DEFAULT_REGISTRATION_STAGE_POLICY,
    DEFAULT_RENDER_PLOTS,
    DEFAULT_SHOW_PLOT,
    DEFAULT_SINGLE_FIGURE_SIZE,
    LOG_LEVEL_CHOICES,
    MEASUREMENT_AREA_MAX_X_MM,
    MEASUREMENT_AREA_MAX_Y_MM,
    MEASUREMENT_AREA_MIN_MM_EXCLUSIVE,
    PlottingConfig,
    WorkflowConfig,
    default_registration_stages,
)


class RegistrationStageSchema(BaseModel):
    translation_step: float = Field(gt=0)
    rot_step_deg: float = Field(ge=0)
    rot_span_deg: float = Field(ge=0)
    tx_steps: int = Field(ge=0)
    ty_steps: int = Field(ge=0)


class PlottingConfigSchema(BaseModel):
    window_mm: tuple[float, float, float, float] = DEFAULT_PLOT_WINDOW_MM
    font_size: float = Field(default=DEFAULT_PLOT_FONT_SIZE, gt=0)
    single_figure_size: tuple[float, float] = DEFAULT_SINGLE_FIGURE_SIZE
    combined_figure_size: tuple[float, float] = DEFAULT_COMBINED_FIGURE_SIZE
    figure_facecolor: str = DEFAULT_PLOT_FIGURE_FACECOLOR
    dark_axes_facecolor: str = DEFAULT_PLOT_DARK_AXES_FACECOLOR
    light_axes_facecolor: str = DEFAULT_PLOT_LIGHT_AXES_FACECOLOR
    save_dpi: int = Field(default=DEFAULT_PLOT_SAVE_DPI, gt=0)
    not_evaluated_color: str = DEFAULT_PLOT_NOT_EVALUATED_COLOR
    measurement_area_x_mm: float | None = None
    measurement_area_y_mm: float | None = None

    @field_validator("single_figure_size", "combined_figure_size")
    @classmethod
    def _validate_figure_size(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("figure sizes must be positive")
        return value


class WorkflowConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measured_file_path: str = DEFAULT_MEASURED_FILE_PATH
    reference_file_path: str = DEFAULT_REFERENCE_FILE_PATH
    power_level_dbm: float = DEFAULT_POWER_LEVEL_DBM

    loaded_images_save_path: str | None = None
    reference_image_save_path: str | None = None
    measured_image_save_path: str | None = None
    aligned_meas_save_path: str | None = None
    registered_image_save_path: str | None = None
    gamma_comparison_image_path: str | None = None

    resample_resolution: float | None = Field(default=None, gt=0)
    noise_floor: float = Field(default=DEFAULT_NOISE_FLOOR, ge=0)
    show_plot: bool = DEFAULT_SHOW_PLOT
    render_plots: bool = DEFAULT_RENDER_PLOTS

    transform_type: Transform2D = Transform2D.RIGID
    stages: list[RegistrationStageSchema] = Field(
        default_factory=lambda: [
            RegistrationStageSchema(**stage) for stage in default_registration_stages()
        ]
    )
    registration_stage_policy: Literal["static", "adaptive"] = (
        DEFAULT_REGISTRATION_STAGE_POLICY
    )
    adaptive_assume_axial_symmetry: bool = DEFAULT_ADAPTIVE_ASSUME_AXIAL_SYMMETRY
    adaptive_max_stages: int = Field(default=DEFAULT_ADAPTIVE_MAX_STAGES, ge=1, le=8)
    adaptive_max_stage_evals: int = Field(
        default=DEFAULT_ADAPTIVE_MAX_STAGE_EVALS, ge=100, le=1_000_000
    )

    dose_to_agreement: float = Field(default=DEFAULT_DOSE_TO_AGREEMENT, gt=0)
    distance_to_agreement: float = Field(default=DEFAULT_DISTANCE_TO_AGREEMENT, gt=0)
    gamma_cap: float = Field(default=DEFAULT_GAMMA_CAP, gt=0)
    evaluation_roi_policy: Literal["reference_only", "intersection", "none"] = (
        DEFAULT_EVALUATION_ROI_POLICY
    )
    save_failures_overlay: bool = True
    log_level: str = DEFAULT_LOG_LEVEL
    plotting: PlottingConfigSchema = Field(default_factory=PlottingConfigSchema)
    output_dir: str | None = None
    measurement_area_x_mm: float | None = Field(
        default=None,
        gt=MEASUREMENT_AREA_MIN_MM_EXCLUSIVE,
        le=MEASUREMENT_AREA_MAX_X_MM,
    )
    measurement_area_y_mm: float | None = Field(
        default=None,
        gt=MEASUREMENT_AREA_MIN_MM_EXCLUSIVE,
        le=MEASUREMENT_AREA_MAX_Y_MM,
    )
    min_inscribed_square_mm: float = Field(
        default=DEFAULT_MIN_INSCRIBED_SQUARE_MM, gt=0
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        allowed = set(LOG_LEVEL_CHOICES)
        if level not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            raise ValueError(f"log_level must be one of: {allowed_str}")
        return level

    @field_validator("measured_file_path", "reference_file_path")
    @classmethod
    def _validate_csv_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path cannot be empty")
        return value

    @model_validator(mode="after")
    def _validate_stage_rotation_fields(self) -> WorkflowConfigSchema:
        if self.transform_type == Transform2D.TRANSLATE:
            return self

        for index, stage in enumerate(self.stages):
            if stage.rot_step_deg <= 0:
                raise ValueError(
                    f"stages.{index}.rot_step_deg must be > 0 for rigid registration"
                )
            if stage.rot_span_deg <= 0:
                raise ValueError(
                    f"stages.{index}.rot_span_deg must be > 0 for rigid registration"
                )
        return self

    @model_validator(mode="after")
    def _validate_measurement_area(self) -> WorkflowConfigSchema:
        x = self.measurement_area_x_mm
        y = self.measurement_area_y_mm
        if (x is None) != (y is None):
            raise ValueError(
                "measurement_area_x_mm and measurement_area_y_mm must be set together"
            )
        if x is not None and y is not None:
            side = max(x, y)
            half = side / 2.0
            self.plotting = self.plotting.model_copy(
                update={
                    "window_mm": (-half, half, -half, half),
                    "measurement_area_x_mm": x,
                    "measurement_area_y_mm": y,
                }
            )
        return self

    def to_workflow_config(self) -> WorkflowConfig:
        data = self.model_dump()
        data["stages"] = [stage.model_dump() for stage in self.stages]
        data["plotting"] = PlottingConfig(**data["plotting"])
        return WorkflowConfig(**data)


def validate_workflow_config(raw: dict[str, Any] | WorkflowConfig) -> WorkflowConfig:
    raw_input = asdict(raw) if isinstance(raw, WorkflowConfig) else raw

    try:
        schema = WorkflowConfigSchema.model_validate(raw_input)
    except ValidationError as exc:
        raise ConfigValidationError(str(exc)) from exc
    return schema.to_workflow_config()


def ensure_input_files_exist(config: WorkflowConfig) -> None:
    missing = [
        p
        for p in (config.measured_file_path, config.reference_file_path)
        if not Path(p).is_file()
    ]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise ConfigValidationError(f"Input file(s) not found: {missing_list}")
