import json
import logging
import os
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from hashlib import md5
from pathlib import Path

import numpy as np
import pytest

from sar_pattern_validation.workflow_config import PlottingConfig, WorkflowConfig
from sar_pattern_validation.workflows import _complete_workflow

from .helpers import compare_gamma_maps

ARTIFACT_DIR = Path(__file__).parent / "artifacts" / "measurement_validation"
LOG_DIR = ARTIFACT_DIR / "logs"
REGENERATE_ENV = "REGENERATE_MEASUREMENT_VALIDATION_ARTIFACTS"
GAMMA_DIFF_THRESHOLD_ENV = "MEASUREMENT_GAMMA_DIFF_THRESHOLD"
SAVE_PLOTS_ENV = "SAVE_MEASUREMENT_VALIDATION_PLOTS"
NOISE_FLOOR_WKG = 0.05
NOISE_FLOOR_LOW_POWER_WKG = 0.01
LOW_POWER_THRESHOLD_DBM = 9
SPACEY_MEASUREMENT_RE = re.compile(
    r"^D(?P<freq>(?:[0-9]+GHz|[0-9]+))_Flat HSL_(?P<distance_mm>[0-9]+) mm_"
    r"(?P<power_dbm>-?[0-9]+) dBm_(?P<mass>1g|10g)_(?P<index>[0-9]+)\.csv$"
)
COMPACT_MEASUREMENT_RE = re.compile(
    r"^D(?P<freq>[0-9]+)_Flat_(?P<distance_mm>[0-9]+)mm_"
    r"(?P<power_dbm>-?[0-9]+)dBm_(?P<mass>1g|10g)_(?P<index>[0-9]+)\.csv$"
)
FREQ_TO_REFERENCE_MHZ = {
    "5GHz": "5800",
}


@dataclass(frozen=True)
class MeasurementValidationCase:
    case_id: str
    measured_csv: str
    reference_csv: str
    power_level_dbm: float
    power_level_key: str
    frequency_key: str
    frequency_label: str
    frequency_mhz: int
    group_key: str
    distance_mm: int
    averaging_mass: str
    source_group: str
    legacy_case_id: str | None = None


@dataclass(frozen=True)
class MeasurementValidationResult:
    actual: dict
    gamma_map: np.ndarray
    evaluation_mask: np.ndarray


@dataclass(frozen=True)
class Artifact:
    artifact_version: int
    dataset: dict
    inputs: dict
    success_thresholds: dict
    expected: dict


def _format_frequency_key(frequency_mhz: int) -> str:
    return f"{frequency_mhz}mhz"


def _format_power_level_key(power_level_dbm: float) -> str:
    if float(power_level_dbm).is_integer():
        value = str(int(power_level_dbm))
    else:
        value = f"{power_level_dbm:g}".replace("-", "neg").replace(".", "p")
    return f"{value}dbm"


def _format_frequency_label(frequency_mhz: int) -> str:
    if frequency_mhz >= 1000:
        return f"{frequency_mhz / 1000:g} GHz"
    return f"{frequency_mhz} MHz"


def _format_group_key(frequency_mhz: int, power_level_dbm: float) -> str:
    return (
        f"{_format_frequency_key(frequency_mhz)}_"
        f"{_format_power_level_key(power_level_dbm)}"
    )


def _make_case(
    *,
    case_id: str,
    measured_csv: str,
    reference_csv: str,
    power_level_dbm: float,
    frequency_mhz: int,
    distance_mm: int,
    averaging_mass: str,
    source_group: str,
    legacy_case_id: str | None = None,
) -> MeasurementValidationCase:
    power_level_key = _format_power_level_key(power_level_dbm)
    return MeasurementValidationCase(
        case_id=case_id,
        measured_csv=measured_csv,
        reference_csv=reference_csv,
        power_level_dbm=power_level_dbm,
        power_level_key=power_level_key,
        frequency_key=_format_frequency_key(frequency_mhz),
        frequency_label=_format_frequency_label(frequency_mhz),
        frequency_mhz=frequency_mhz,
        group_key=_format_group_key(frequency_mhz, power_level_dbm),
        distance_mm=distance_mm,
        averaging_mass=averaging_mass,
        source_group=source_group,
        legacy_case_id=legacy_case_id,
    )


BASELINE_CASES = tuple(
    _make_case(
        case_id=f"2450_10mm_1g_17dbm_{i + 1}",
        measured_csv=f"data/measurements/dipole_2450MHz_Flat_10mm_17dBm_1g_{i + 1}.csv",
        reference_csv="data/database/dipole_2450MHz_Flat_10mm_1g.csv",
        power_level_dbm=17.0,
        frequency_mhz=2450,
        distance_mm=10,
        averaging_mass="1g",
        source_group="baseline",
    )
    for i in range(9)
)

ROBUSTNESS_CASES = (
    _make_case(
        case_id="900_15mm_1g_sparse_rotation",
        measured_csv="data/measurements/D900_Flat HSL_15 mm_10 dBm_1g_10.csv",
        reference_csv="data/database/dipole_900MHz_Flat_15mm_1g.csv",
        power_level_dbm=10.0,
        frequency_mhz=900,
        distance_mm=15,
        averaging_mass="1g",
        source_group="robustness",
    ),
    _make_case(
        case_id="1950_10mm_10g_noisy",
        measured_csv="data/measurements/D1950_Flat HSL_10 mm_4 dBm_10g_14.csv",
        reference_csv="data/database/dipole_1950MHz_Flat_10mm_10g.csv",
        power_level_dbm=4.0,
        frequency_mhz=1950,
        distance_mm=10,
        averaging_mass="10g",
        source_group="robustness",
    ),
    _make_case(
        case_id="5ghz_10mm_1g_noisy",
        measured_csv="data/measurements/D5GHz_Flat HSL_10 mm_1 dBm_1g_15.csv",
        reference_csv="data/database/dipole_5800MHz_Flat_10mm_1g.csv",
        power_level_dbm=1.0,
        frequency_mhz=5800,
        distance_mm=10,
        averaging_mass="1g",
        source_group="robustness",
    ),
)


def _case_from_measurement_filename(filename: str) -> MeasurementValidationCase:
    match = SPACEY_MEASUREMENT_RE.match(filename) or COMPACT_MEASUREMENT_RE.match(
        filename
    )
    if not match:
        raise ValueError(f"Unsupported measurement filename format: {filename}")

    freq_token = match.group("freq")
    freq_mhz = int(FREQ_TO_REFERENCE_MHZ.get(freq_token, freq_token))
    distance_mm = match.group("distance_mm")
    mass = match.group("mass")
    power_dbm = float(match.group("power_dbm"))
    idx = match.group("index")
    normalized_freq_token = freq_token.lower()
    case_id = (
        f"{normalized_freq_token}_{distance_mm}mm_{mass}_{int(power_dbm)}dbm_{idx}"
    )

    return _make_case(
        case_id=case_id,
        measured_csv=f"data/measurements/{filename}",
        reference_csv=f"data/database/dipole_{freq_mhz}MHz_Flat_{distance_mm}mm_{mass}.csv",
        power_level_dbm=power_dbm,
        frequency_mhz=freq_mhz,
        distance_mm=int(distance_mm),
        averaging_mass=mass,
        source_group="expanded_measurements",
        legacy_case_id=f"zip_{case_id}",
    )


def _discover_measurement_cases() -> tuple[MeasurementValidationCase, ...]:
    already_included = {case.measured_csv for case in BASELINE_CASES + ROBUSTNESS_CASES}
    candidates = sorted(Path("data/measurements").glob("D*.csv"), key=lambda p: p.name)

    def _file_signature(path: str) -> str:
        hasher = md5(usedforsecurity=False)
        with Path(path).open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    seen_signatures: set[tuple[str, float, str]] = set()
    for existing_case in BASELINE_CASES + ROBUSTNESS_CASES:
        measured_path = Path(existing_case.measured_csv)
        if measured_path.exists():
            seen_signatures.add(
                (
                    existing_case.reference_csv,
                    existing_case.power_level_dbm,
                    _file_signature(existing_case.measured_csv),
                )
            )

    discovered: list[MeasurementValidationCase] = []
    for measured_path in candidates:
        measured_csv = f"data/measurements/{measured_path.name}"
        if measured_csv in already_included:
            continue

        case = _case_from_measurement_filename(measured_path.name)
        if not Path(case.reference_csv).exists():
            raise FileNotFoundError(
                f"Reference SAR database file not found for {measured_path.name}: "
                f"{case.reference_csv}"
            )

        signature = (
            case.reference_csv,
            case.power_level_dbm,
            _file_signature(case.measured_csv),
        )
        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        discovered.append(case)

    return tuple(discovered)


DISCOVERED_CASES = _discover_measurement_cases()

CASES = BASELINE_CASES + ROBUSTNESS_CASES + DISCOVERED_CASES


def _build_cases_by_group(
    cases: tuple[MeasurementValidationCase, ...],
) -> dict[str, tuple[MeasurementValidationCase, ...]]:
    grouped: dict[str, list[MeasurementValidationCase]] = defaultdict(list)
    for case in cases:
        grouped[case.group_key].append(case)

    return {
        group_key: tuple(
            sorted(
                grouped_cases,
                key=lambda case: (
                    case.distance_mm,
                    case.power_level_dbm,
                    case.averaging_mass,
                    case.case_id,
                ),
            )
        )
        for group_key, grouped_cases in sorted(
            grouped.items(),
            key=lambda entry: (
                entry[1][0].frequency_mhz,
                entry[1][0].power_level_dbm,
            ),
        )
    }


CASES_BY_GROUP = _build_cases_by_group(CASES)


def _case_group_dir(case: MeasurementValidationCase) -> Path:
    return ARTIFACT_DIR / case.frequency_key / case.power_level_key


def _artifact_json_path(case: MeasurementValidationCase) -> Path:
    return _case_group_dir(case) / f"{case.case_id}_metrics.json"


def _artifact_gamma_path(case: MeasurementValidationCase) -> Path:
    return _case_group_dir(case) / f"{case.case_id}_gamma_field.npz"


def _legacy_artifact_json_path(case: MeasurementValidationCase) -> Path | None:
    if case.legacy_case_id is None:
        return None
    return _case_group_dir(case) / f"{case.legacy_case_id}_metrics.json"


def _legacy_artifact_gamma_path(case: MeasurementValidationCase) -> Path | None:
    if case.legacy_case_id is None:
        return None
    return _case_group_dir(case) / f"{case.legacy_case_id}_gamma_field.npz"


def _artifact_json_candidate_paths(case: MeasurementValidationCase) -> tuple[Path, ...]:
    candidates = [
        _artifact_json_path(case),
        ARTIFACT_DIR / case.frequency_key / f"{case.case_id}_metrics.json",
        ARTIFACT_DIR / f"{case.case_id}_metrics.json",
    ]
    legacy_path = _legacy_artifact_json_path(case)
    if legacy_path is not None:
        candidates.append(legacy_path)
        candidates.append(
            ARTIFACT_DIR / case.frequency_key / f"{case.legacy_case_id}_metrics.json"
        )
        candidates.append(ARTIFACT_DIR / f"{case.legacy_case_id}_metrics.json")
    return tuple(candidates)


def _artifact_gamma_candidate_paths(
    case: MeasurementValidationCase,
) -> tuple[Path, ...]:
    candidates = [
        _artifact_gamma_path(case),
        ARTIFACT_DIR / case.frequency_key / f"{case.case_id}_gamma_field.npz",
        ARTIFACT_DIR / f"{case.case_id}_gamma_field.npz",
    ]
    legacy_path = _legacy_artifact_gamma_path(case)
    if legacy_path is not None:
        candidates.append(legacy_path)
        candidates.append(
            ARTIFACT_DIR / case.frequency_key / f"{case.legacy_case_id}_gamma_field.npz"
        )
        candidates.append(ARTIFACT_DIR / f"{case.legacy_case_id}_gamma_field.npz")
    return tuple(candidates)


def _first_existing_path(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _case_noise_floor(case: MeasurementValidationCase) -> float:
    return (
        NOISE_FLOOR_LOW_POWER_WKG
        if case.power_level_dbm <= LOW_POWER_THRESHOLD_DBM
        else NOISE_FLOOR_WKG
    )


def _artifact_payload(case: MeasurementValidationCase, actual: dict) -> dict:
    return {
        "artifact_version": 1,
        "dataset": {
            "measured_csv": case.measured_csv,
            "reference_csv": case.reference_csv,
            "group_key": case.group_key,
            "frequency_key": case.frequency_key,
            "frequency_label": case.frequency_label,
            "frequency_mhz": case.frequency_mhz,
            "power_level_key": case.power_level_key,
            "power_level_dbm": case.power_level_dbm,
            "distance_mm": case.distance_mm,
            "averaging_mass": case.averaging_mass,
            "source_group": case.source_group,
        },
        "inputs": {
            "power_level_dbm": case.power_level_dbm,
            "noise_floor_wkg": _case_noise_floor(case),
            "dose_to_agreement_percent": 10.0,
            "distance_to_agreement_mm": 3.0,
            "gamma_cap": 2.0,
            "evaluation_roi_policy": "intersection",
        },
        "success_thresholds": {
            "require_zero_failed_pixels": True,
        },
        "expected": actual,
    }


def _case_plot_dir(case: MeasurementValidationCase) -> Path:
    return (
        ARTIFACT_DIR
        / "plots"
        / case.frequency_key
        / case.power_level_key
        / case.case_id
    )


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _case_debug_log_path(case: MeasurementValidationCase, test_name: str) -> Path:
    freq_log_dir = LOG_DIR / case.frequency_key / case.power_level_key
    freq_log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_filename(test_name)
    return freq_log_dir / f"{timestamp}_{safe_name}_{case.case_id}.log"


@contextmanager
def _debug_file_logging(log_path: Path):
    root = logging.getLogger()
    prev_root_level = root.level

    reg_logger = logging.getLogger("Rigid2DRegistration")
    prev_reg_level = reg_logger.level
    matplotlib_logger = logging.getLogger("matplotlib")
    prev_matplotlib_level = matplotlib_logger.level

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    reg_logger.setLevel(logging.DEBUG)
    matplotlib_logger.setLevel(logging.WARNING)

    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(prev_root_level)
        reg_logger.setLevel(prev_reg_level)
        matplotlib_logger.setLevel(prev_matplotlib_level)


def _write_artifacts(
    case: MeasurementValidationCase, result: MeasurementValidationResult
) -> None:
    _case_group_dir(case).mkdir(parents=True, exist_ok=True)
    _artifact_json_path(case).write_text(
        json.dumps(_artifact_payload(case, result.actual), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        _artifact_gamma_path(case),
        gamma_map=result.gamma_map.astype(np.float32),
        evaluation_mask=result.evaluation_mask.astype(np.uint8),
    )


def _load_artifact(case: MeasurementValidationCase) -> Artifact:
    artifact_path = _first_existing_path(_artifact_json_candidate_paths(case))
    assert artifact_path is not None, f"missing artifact json for {case.case_id}"
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    return Artifact(**data)


def _load_gamma_field(case: MeasurementValidationCase) -> tuple[np.ndarray, np.ndarray]:
    artifact_path = _first_existing_path(_artifact_gamma_candidate_paths(case))
    assert artifact_path is not None, f"missing gamma artifact for {case.case_id}"
    data = np.load(artifact_path)
    return data["gamma_map"], data["evaluation_mask"]


def _case_report_payload(
    case: MeasurementValidationCase, actual: dict, log_path: Path
) -> dict:
    return {
        "case_id": case.case_id,
        "legacy_case_id": case.legacy_case_id,
        "group_key": case.group_key,
        "frequency_key": case.frequency_key,
        "frequency_label": case.frequency_label,
        "frequency_mhz": case.frequency_mhz,
        "power_level_key": case.power_level_key,
        "distance_mm": case.distance_mm,
        "averaging_mass": case.averaging_mass,
        "power_level_dbm": case.power_level_dbm,
        "source_group": case.source_group,
        "measured_csv": case.measured_csv,
        "reference_csv": case.reference_csv,
        "artifact_metrics_path": str(_artifact_json_path(case)),
        "artifact_gamma_path": str(_artifact_gamma_path(case)),
        "artifact_metrics_search_paths": [
            str(path) for path in _artifact_json_candidate_paths(case)
        ],
        "artifact_gamma_search_paths": [
            str(path) for path in _artifact_gamma_candidate_paths(case)
        ],
        "log_path": str(log_path),
        "quality_gate_zero_failed_pixels": actual["failed_pixel_count"] == 0,
        **actual,
    }


def _compute_case(
    case: MeasurementValidationCase, *, save_plots: bool
) -> MeasurementValidationResult:
    plot_dir = _case_plot_dir(case)
    config = WorkflowConfig(
        measured_file_path=case.measured_csv,
        reference_file_path=case.reference_csv,
        power_level_dbm=case.power_level_dbm,
        log_level="DEBUG",
        noise_floor=_case_noise_floor(case),
        dose_to_agreement=10.0,
        distance_to_agreement=3.0,
        gamma_cap=2.0,
        registration_stage_policy="adaptive",
        adaptive_assume_axial_symmetry=True,
        evaluation_roi_policy="intersection",
        render_plots=save_plots,
        plotting=PlottingConfig(save_colorbars=False),
        loaded_images_save_path=str(plot_dir / "01_loader_comparison.png")
        if save_plots
        else None,
        aligned_meas_save_path=str(plot_dir / "02_registered_measured.png")
        if save_plots
        else None,
        registered_image_save_path=str(plot_dir / "02_registration_overlay.png")
        if save_plots
        else None,
        gamma_comparison_image_path=str(plot_dir / "03_gamma_map.png")
        if save_plots
        else None,
        save_failures_overlay=save_plots,
    )
    wf = _complete_workflow(config)

    assert wf.gamma_map is not None
    assert wf.evaluation_mask is not None

    actual = {
        "evaluated_pixel_count": wf.evaluated_pixel_count,
        "passed_pixel_count": wf.passed_pixel_count,
        "failed_pixel_count": wf.failed_pixel_count,
        "pass_rate_percent": wf.pass_rate_percent,
        "measured_pssar": wf.measured_pssar,
        "reference_pssar": wf.reference_pssar,
        "scaling_error": wf.scaling_error,
    }
    return MeasurementValidationResult(
        actual=actual,
        gamma_map=wf.gamma_map.astype(np.float32),
        evaluation_mask=wf.evaluation_mask.astype(np.uint8),
    )


def _assert_measurement_case_matches_reference_artifacts(
    measurement_case: MeasurementValidationCase,
    request: pytest.FixtureRequest,
) -> None:
    case = measurement_case
    save_plots = os.getenv(SAVE_PLOTS_ENV, "1") == "1"
    log_path = _case_debug_log_path(case, request.node.name)
    with _debug_file_logging(log_path):
        result = _compute_case(case, save_plots=save_plots)

    actual = result.actual
    request.node.user_properties.append(
        (
            "measurement_validation_case",
            json.dumps(_case_report_payload(case, actual, log_path), sort_keys=True),
        )
    )
    assert actual["failed_pixel_count"] == 0, (
        f"{case.case_id} expected zero failed pixels, got "
        f"{actual['failed_pixel_count']} out of {actual['evaluated_pixel_count']}"
    )

    if os.getenv(REGENERATE_ENV) == "1":
        _write_artifacts(case, result)
        return
    else:
        # Fail fast if artifacts don't exist
        metrics_path = _first_existing_path(_artifact_json_candidate_paths(case))
        gamma_path = _first_existing_path(_artifact_gamma_candidate_paths(case))

        if metrics_path is None or gamma_path is None:
            pytest.fail(
                f"Artifacts missing for {case.case_id}:\n"
                f"  Metrics: {metrics_path is not None}\n"
                f"  Gamma:   {gamma_path is not None}\n"
                f"  Expected directory: {_case_group_dir(case)}\n"
                f"  Run with --regenerate-artifacts or use run_measurement_validation_tests.py"
            )

        artifact = _load_artifact(case)
        assert artifact.dataset["measured_csv"] == case.measured_csv
        assert artifact.dataset["reference_csv"] == case.reference_csv
        assert artifact.inputs["evaluation_roi_policy"] == "intersection"
        assert (
            artifact.success_thresholds.get("require_zero_failed_pixels", True) is True
        )

        expected = artifact.expected
        assert expected is not None, f"{case.case_id} artifact missing expected results"
        assert actual["evaluated_pixel_count"] == expected["evaluated_pixel_count"]
        assert actual["passed_pixel_count"] == expected["passed_pixel_count"]
        assert actual["failed_pixel_count"] == expected["failed_pixel_count"]
        assert actual["pass_rate_percent"] == pytest.approx(
            expected["pass_rate_percent"], abs=1e-9
        )
        assert actual["measured_pssar"] == pytest.approx(
            expected["measured_pssar"], abs=1e-9
        )
        assert actual["reference_pssar"] == pytest.approx(
            expected["reference_pssar"], abs=1e-9
        )
        assert actual["scaling_error"] == pytest.approx(
            expected["scaling_error"], abs=1e-9
        )

        expected_gamma, expected_mask = _load_gamma_field(case)
        assert expected_gamma is not None, (
            f"{case.case_id} artifact missing expected gamma field"
        )
        assert expected_mask is not None, (
            f"{case.case_id} artifact missing expected mask"
        )

        compare_gamma_maps(
            expected_gamma, expected_mask, result.gamma_map, result.evaluation_mask
        )


def _build_group_test(group_key: str, cases: tuple[MeasurementValidationCase, ...]):
    @pytest.mark.slow
    @pytest.mark.parametrize("measurement_case", cases, ids=lambda case: case.case_id)
    def _test(
        measurement_case: MeasurementValidationCase,
        request: pytest.FixtureRequest,
    ) -> None:
        _assert_measurement_case_matches_reference_artifacts(measurement_case, request)

    _test.__name__ = (
        f"test_measurement_workflow_cases_{group_key}_match_reference_artifacts"
    )
    return _test


for _group_key, _group_cases in CASES_BY_GROUP.items():
    globals()[
        f"test_measurement_workflow_cases_{_group_key}_match_reference_artifacts"
    ] = _build_group_test(_group_key, _group_cases)


del _group_key
del _group_cases
