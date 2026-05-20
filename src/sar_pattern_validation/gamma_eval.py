from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion

from sar_pattern_validation.plotting import plot_gamma_results
from sar_pattern_validation.utils import image_extent_mm
from sar_pattern_validation.workflow_config import PlottingConfig


class GammaMapEvaluator:
    """
    Gamma evaluation for 2D SAR maps (paper-aligned, linear space)

    Inputs
    ------
    reference_sar_linear :
        sitk.Image, peak-normalized linear SAR on the reference grid.

    measured_sar_linear :
        sitk.Image, peak-normalized linear SAR on the measured grid.

    reference_to_measured_transform :
        sitk.Transform mapping reference coordinates -> measured coordinates.
        This is the same mapping used to build the overlay (reference warped onto
        the measured grid). Gamma is evaluated in the measured frame.

    Gamma definition (abstract)
    ---------------------------
    Each evaluated point (in the measured frame) is compared to all reference points
    within a search neighborhood to find the minimum:

        Γ^2 = (Δr / Δd)^2 + (ΔSAR / ΔD)^2

    where:
      Δd : distance-to-agreement (mm)
      ΔD : dose-to-agreement (fraction of peak, because inputs are peak-normalized)

    Because both inputs are peak-normalized (peak = 1):
        ΔD_linear = dose_to_agreement_percent / 100

    ROI / masking
    -------------
    The abstract masks values before normalization based on an absolute threshold.
    Those masks are supplied here (optional) and are used only to define the evaluation region.

    On the measured grid:
        evaluation_roi = measured_mask  ∩  resampled(reference_mask)

    If no masks are provided, evaluation covers all pixels.
    """

    def __init__(
        self,
        *,
        reference_sar_linear: sitk.Image,
        measured_sar_linear: sitk.Image,
        reference_to_measured_transform: sitk.Transform,
        dose_to_agreement_percent: float = 5.0,  # ΔD in % of peak (peak-normalized inputs)
        distance_to_agreement_mm: float = 2.0,  # Δd in mm
        gamma_cap: float = 2.0,
    ):
        self.reference_sar_linear = sitk.Cast(reference_sar_linear, sitk.sitkFloat32)
        self.measured_sar_linear = sitk.Cast(measured_sar_linear, sitk.sitkFloat32)
        self.reference_to_measured_transform = reference_to_measured_transform

        self.dose_to_agreement_percent = float(dose_to_agreement_percent)
        self.distance_to_agreement_mm = float(distance_to_agreement_mm)
        self.gamma_cap = float(gamma_cap)

        self.reference_mask_u8: sitk.Image | None = None
        self.measured_mask_u8: sitk.Image | None = None

        self.gamma_map: np.ndarray | None = None
        self.evaluation_mask: np.ndarray | None = None
        self.pass_rate_percent: float | None = None
        self.evaluated_pixel_count: int | None = None
        self.passed_pixel_count: int | None = None
        self.failed_pixel_count: int | None = None

    # ------------------------------ public ------------------------------

    def compute(self) -> None:
        """
        Compute gamma on the measured grid.

        Output fields
        -------------
        gamma_map :
            float32 array (H,W) with NaN outside evaluation_mask (if masks provided).

        evaluation_mask :
            bool array (H,W) indicating where gamma is evaluated.

        pass_rate_percent :
            100 * (# pixels with gamma <= 1) / (# evaluated pixels)
        """
        reference_on_measured = self._resample_reference_onto_measured()

        measured_arr = sitk.GetArrayFromImage(self.measured_sar_linear).astype(
            np.float32
        )
        reference_arr = sitk.GetArrayFromImage(reference_on_measured).astype(np.float32)

        distance_to_agreement_pixels = self._mm_to_pixels(
            self.distance_to_agreement_mm, self.measured_sar_linear
        )
        dose_to_agreement_fraction = max(self.dose_to_agreement_percent / 100.0, 1e-12)

        evaluation_mask = self._build_evaluation_mask_on_measured()
        gamma = self._gamma_2d_peak_normalized(
            reference=reference_arr,
            evaluation=measured_arr,
            distance_to_agreement_pixels=distance_to_agreement_pixels,
            dose_to_agreement_fraction=dose_to_agreement_fraction,
            gamma_cap=self.gamma_cap,
        )

        if evaluation_mask is not None:
            gamma[~evaluation_mask] = np.nan
            evaluated_mask = evaluation_mask
        else:
            evaluated_mask = ~np.isnan(gamma)

        evaluated_pixel_count = int(np.sum(evaluated_mask))
        passed_pixel_count = int(np.sum((gamma <= 1.0) & evaluated_mask))
        failed_pixel_count = max(0, evaluated_pixel_count - passed_pixel_count)

        self.gamma_map = gamma
        self.evaluation_mask = evaluated_mask
        self.evaluated_pixel_count = evaluated_pixel_count
        self.passed_pixel_count = passed_pixel_count
        self.failed_pixel_count = failed_pixel_count
        self.pass_rate_percent = (
            100.0 * passed_pixel_count / max(evaluated_pixel_count, 1)
        )

    def show(
        self,
        *,
        gamma_image_save_path: Path | None = None,
        failure_image_save_path: Path | None = None,
        noise_floor_mask: np.ndarray | None = None,
        plotting_config: PlottingConfig | None = None,
    ) -> None:
        """
        Display gamma map and failures (gamma > 1) on the measured grid.
        """
        if (
            self.gamma_map is None
            or self.evaluation_mask is None
            or self.pass_rate_percent is None
            or self.evaluated_pixel_count is None
            or self.passed_pixel_count is None
            or self.failed_pixel_count is None
        ):
            raise RuntimeError("compute() must be called before show().")

        plot_gamma_results(
            gamma_map=self.gamma_map,
            evaluation_mask=self.evaluation_mask,
            gamma_cap=self.gamma_cap,
            extent_mm=self._extent_mm(self.measured_sar_linear),
            gamma_image_save_path=gamma_image_save_path,
            failure_image_save_path=failure_image_save_path,
            noise_floor_mask=noise_floor_mask,
            plotting_config=plotting_config,
        )

    def _resample_reference_onto_measured(self) -> sitk.Image:
        """
        Resample reference_sar_linear onto the measured grid using
        reference->measured transform.

        Outside-domain default is 0.0 (handled by ROI masking if masks are provided).
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.measured_sar_linear)
        resampler.SetTransform(self.reference_to_measured_transform.GetInverse())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        return resampler.Execute(self.reference_sar_linear)

    def _build_evaluation_mask_on_measured(self) -> np.ndarray | None:
        """
        evaluation_mask = measured_mask ∩ resampled(reference_mask) on the measured grid.

        Returns None when neither mask is provided.
        """
        measured_roi = None
        if self.measured_mask_u8 is not None:
            measured_roi = sitk.GetArrayFromImage(self.measured_mask_u8).astype(bool)

        reference_roi_on_measured = None
        if self.reference_mask_u8 is not None:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(self.measured_sar_linear)
            resampler.SetTransform(self.reference_to_measured_transform.GetInverse())
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            reference_mask_on_meas = resampler.Execute(self.reference_mask_u8)
            reference_roi_on_measured = sitk.GetArrayFromImage(
                reference_mask_on_meas
            ).astype(bool)

        if measured_roi is None and reference_roi_on_measured is None:
            return None
        if measured_roi is None:
            return reference_roi_on_measured
        if reference_roi_on_measured is None:
            return measured_roi
        return measured_roi & reference_roi_on_measured

    @staticmethod
    def _mm_to_pixels(distance_mm: float, img: sitk.Image) -> int:
        sx_m, sy_m = img.GetSpacing()
        min_spacing_m = max(min(float(sx_m), float(sy_m)), 1e-12)
        radius_m = float(distance_mm) / 1000.0
        return max(1, int(round(radius_m / min_spacing_m)))

    @staticmethod
    def _gamma_2d_peak_normalized(
        *,
        reference: np.ndarray,
        evaluation: np.ndarray,
        distance_to_agreement_pixels: int,
        dose_to_agreement_fraction: float,
        gamma_cap: float,
    ) -> np.ndarray:
        """
        Gamma for peak-normalized linear SAR (peak=1).

        Spatial term:
            (Δr / Δd_pixels)^2

        Dose term:
            (ΔSAR / ΔD_fraction)^2
        """
        if reference.shape != evaluation.shape or reference.ndim != 2:
            raise ValueError(
                "reference and evaluation must be 2D arrays with matching shape."
            )

        dta = int(distance_to_agreement_pixels)
        dose_tol = float(max(dose_to_agreement_fraction, 1e-12))
        search_radius = max(1, int(np.ceil(float(gamma_cap) * dta)))

        reference_padded = np.pad(reference, search_radius, mode="edge")

        offsets = np.arange(-search_radius, search_radius + 1, dtype=np.int32)
        rr_grid, cc_grid = np.meshgrid(offsets, offsets, indexing="ij")
        within_radius = (rr_grid**2 + cc_grid**2) <= search_radius**2
        rr = rr_grid[within_radius]
        cc = cc_grid[within_radius]
        dist_pix_sq = rr.astype(np.float32) ** 2 + cc.astype(np.float32) ** 2
        spatial_term_sq = dist_pix_sq / float(dta * dta)

        h, w = reference.shape
        finite_evaluation = np.isfinite(evaluation)

        # Build (N_offsets, H, W) stack of shifted reference views, then vectorize.
        n_offsets = len(rr)
        shifts = np.empty((n_offsets, h, w), dtype=np.float32)
        for i, (dr, dc) in enumerate(zip(rr, cc, strict=False)):
            shifts[i] = reference_padded[
                search_radius + dr : search_radius + dr + h,
                search_radius + dc : search_radius + dc + w,
            ]

        # dose term: (N, H, W)  — suppress invalid where evaluation is non-finite
        with np.errstate(invalid="ignore"):
            dose_term_sq = ((evaluation[None] - shifts) / dose_tol) ** 2

        # spatial term: (N, 1, 1)
        spatial_term_sq_bc = spatial_term_sq.astype(np.float32)[:, None, None]

        # candidate_sq: (N, H, W)
        candidate_sq = spatial_term_sq_bc + dose_term_sq

        # min over offsets -> (H, W)
        gamma_sq = np.min(candidate_sq, axis=0)

        # clamp to cap, sqrt, apply NaN mask
        gamma_cap_sq = float(gamma_cap * gamma_cap)
        np.clip(gamma_sq, None, gamma_cap_sq, out=gamma_sq)
        gamma = np.sqrt(gamma_sq).astype(np.float32, copy=False)
        gamma[~finite_evaluation] = np.nan
        return gamma

    @staticmethod
    def _extent_mm(img: sitk.Image) -> tuple[float, float, float, float]:
        return image_extent_mm(img)

    def evaluation_mask_fits_axis_aligned_square_mm(self, side_mm: float) -> bool:
        """
        Return True iff an axis-aligned square of `side_mm` physical extent
        (in mm) fits entirely within the gamma evaluation mask, without
        rotation. Uses the measured-grid spacing (gamma is evaluated on the
        measured frame after Task 6.1).

        Per MGD 2026-04-24 feedback (slide 7): a 22 mm × 22 mm square — the
        face of the 10 g averaging cube — must fit inside the mask. Per-axis
        bounding-box checks are insufficient (an L-shaped mask can pass them
        without admitting any inscribed square).
        """
        if self.evaluation_mask is None:
            raise RuntimeError(
                "compute() must be called before evaluation_mask_fits_axis_aligned_square_mm()."
            )
        return _mask_fits_axis_aligned_square_mm(
            mask=self.evaluation_mask,
            side_mm=side_mm,
            spacing_m=self.measured_sar_linear.GetSpacing(),
        )


def _mask_fits_axis_aligned_square_mm(
    *,
    mask: np.ndarray,
    side_mm: float,
    spacing_m: tuple[float, float],
) -> bool:
    """
    Test whether an axis-aligned square of `side_mm` × `side_mm` physical
    extent fits entirely inside `mask`.

    Implementation: erode the mask with a rectangular structuring element of
    pixel dimensions large enough to span `side_mm` along each axis (rounded
    up to ensure the physical extent is at least `side_mm`). If any pixel
    survives, a position exists where the structuring element fits entirely
    inside the mask, i.e. the inscribed square test passes.
    """
    sx_mm = float(spacing_m[0]) * 1000.0
    sy_mm = float(spacing_m[1]) * 1000.0
    if sx_mm <= 0 or sy_mm <= 0 or side_mm <= 0:
        return False

    width_px = max(1, int(np.ceil(side_mm / sx_mm)))
    height_px = max(1, int(np.ceil(side_mm / sy_mm)))

    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape[0] < height_px or mask_bool.shape[1] < width_px:
        return False

    structure = np.ones((height_px, width_px), dtype=bool)
    eroded = binary_erosion(mask_bool, structure=structure)
    return bool(eroded.any())
