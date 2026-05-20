import json
import os
from pathlib import Path

import numpy as np
import pytest

from sar_pattern_validation.gamma_eval import GammaMapEvaluator
from sar_pattern_validation.image_loader import SARImageLoader
from sar_pattern_validation.plotting import show_registration_overlay
from sar_pattern_validation.registration2d import Rigid2DRegistration, Transform2D
from sar_pattern_validation.workflows import _apply_roi_policy

from .helpers import compare_gamma_maps

ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "tutorial_reference_metrics.json"
GAMMA_FIELD_PATH = Path(__file__).parent / "artifacts" / "tutorial_gamma_field.npz"
PLOT_DIR = Path(__file__).parent / "artifacts" / "tutorial_plots"
SAVE_PLOTS_ENV = "SAVE_TUTORIAL_VALIDATION_PLOTS"


def _load_artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def _write_artifact(payload: dict) -> None:
    ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_gamma_field() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(GAMMA_FIELD_PATH)
    return data["gamma_map"], data["evaluation_mask"]


def _write_gamma_field(gamma_map: np.ndarray, evaluation_mask: np.ndarray) -> None:
    np.savez_compressed(
        GAMMA_FIELD_PATH,
        gamma_map=gamma_map.astype(np.float32),
        evaluation_mask=evaluation_mask.astype(np.uint8),
    )


@pytest.mark.validation
def test_tutorial_dataset_metrics_match_reference_artifact() -> None:
    artifact = _load_artifact()
    dataset = artifact["dataset"]
    inputs = artifact["inputs"]
    expected = artifact["expected"]
    assert inputs["evaluation_roi_policy"] == "intersection"

    loader = SARImageLoader(
        measured_path=dataset["measured_csv"],
        reference_path=dataset["reference_csv"],
        show_plot=False,
        resample_resolution=inputs["resample_resolution"],
        noise_floor_wkg=inputs["noise_floor_wkg"],
        warn=True,
    )

    reference_db, measured_db = loader.get_images()
    measured_mask_u8, reference_mask_u8 = loader.make_metric_masks()
    measured_support_u8, _ = loader.make_support_masks()

    save_plots = os.getenv(SAVE_PLOTS_ENV, "1") == "1"
    if save_plots:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        loader.plot(
            image_save_path=PLOT_DIR / "01_loader_comparison.png",
            measured_save_path=PLOT_DIR / "01_loader_measured.png",
            reference_save_path=PLOT_DIR / "01_loader_reference.png",
        )

    registration = Rigid2DRegistration(
        fixed_image=measured_db,
        moving_image=reference_db,
        transform_type=Transform2D(inputs["transform_type"]),
    )
    aligned_image, reference_to_measured_tx = registration.run(
        stages=inputs["registration_stages"],
        fixed_mask=None,
        moving_mask=reference_mask_u8,
    )

    if save_plots:
        loader.plot_aligned(
            aligned_image,
            aligned_meas_save_path=PLOT_DIR / "02_registered_simulated.png",
        )
        show_registration_overlay(
            fixed_image=measured_db,
            aligned_moving_image=aligned_image,
            image_save_path=PLOT_DIR / "02_registration_overlay.png",
            title="Registration Overlay (tutorial)",
        )

    evaluator = GammaMapEvaluator(
        reference_sar_linear=loader.reference_image_linear,
        measured_sar_linear=loader.measured_image_linear,
        reference_to_measured_transform=reference_to_measured_tx,
        dose_to_agreement_percent=inputs["dose_to_agreement_percent"],
        distance_to_agreement_mm=inputs["distance_to_agreement_mm"],
        gamma_cap=inputs["gamma_cap"],
    )
    roi_policy = "intersection"
    _apply_roi_policy(
        evaluator,
        reference_mask_u8=reference_mask_u8,
        measured_mask_u8=measured_support_u8,
        policy=roi_policy,
    )
    evaluator.compute()

    if save_plots:
        evaluator.show(
            gamma_image_save_path=PLOT_DIR / "03_gamma_map.png",
            failure_image_save_path=PLOT_DIR / "03_gamma_failures.png",
        )

    actual = {
        "evaluated_pixel_count": evaluator.evaluated_pixel_count,
        "passed_pixel_count": evaluator.passed_pixel_count,
        "failed_pixel_count": evaluator.failed_pixel_count,
        "pass_rate_percent": evaluator.pass_rate_percent,
    }
    assert evaluator.gamma_map is not None
    assert evaluator.evaluation_mask is not None

    if os.getenv("REGENERATE_TUTORIAL_ARTIFACT") == "1":
        artifact.setdefault("inputs", {})["evaluation_roi_policy"] = roi_policy
        artifact["expected"] = actual
        _write_artifact(artifact)
        _write_gamma_field(evaluator.gamma_map, evaluator.evaluation_mask)
        return

    assert actual["evaluated_pixel_count"] == expected["evaluated_pixel_count"]
    assert actual["passed_pixel_count"] == expected["passed_pixel_count"]
    assert actual["failed_pixel_count"] == expected["failed_pixel_count"]
    assert actual["pass_rate_percent"] == pytest.approx(
        expected["pass_rate_percent"], abs=1e-9
    )

    expected_gamma, expected_mask = _load_gamma_field()
    actual_gamma = evaluator.gamma_map
    actual_mask = evaluator.evaluation_mask.astype(np.uint8)

    compare_gamma_maps(expected_gamma, expected_mask, actual_gamma, actual_mask)
