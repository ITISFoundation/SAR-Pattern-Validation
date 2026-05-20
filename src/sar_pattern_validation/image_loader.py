"""
SAR Image Loader
================
Reads measured and reference SAR distributions from CSV files, builds 2D grids,
and produces SimpleITK images suitable for registration and gamma evaluation.

Pipeline
--------
1. Parse CSV → (x, y, SAR) in meters / W·kg⁻¹.
2. Grid the point data (native or resampled to a uniform step).
3. Build absolute masks:  mask = (SAR ≥ cutoff) ∧ support,
   where cutoff = min(threshold_cap, 2 × noise_floor).
4. Peak-normalize each image by its masked peak → linear images (0 outside support).
5. Derive dB images with a shared floor so both live on a compatible dynamic range.

Terminology
-----------
- *reference*: fixed / target grid (orientation preserved).
- *measured*:  moving image (warped onto reference during registration).
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.interpolate import griddata

from sar_pattern_validation.errors import CsvFormatError
from sar_pattern_validation.plotting import plot_loaded_images, plot_sar_image
from sar_pattern_validation.workflow_config import PlottingConfig


class Grid(NamedTuple):
    sar_grid: np.ndarray
    support_mask: np.ndarray
    x_axis_m: np.ndarray
    y_axis_m: np.ndarray
    dx_m: float
    dy_m: float

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.x_axis_m, self.y_axis_m)

    @property
    def spacing(self) -> tuple[float, float]:
        return (self.dx_m, self.dy_m)


class SARImageLoader:
    def __init__(
        self,
        measured_path: str,
        reference_path: str,
        *,
        power_level_dbm: float = 30.0,
        noise_floor_wkg: float = 0.05,
        resample_resolution: float | None = None,
        show_plot: bool = False,
        image_save_path: Path | None = None,
        reference_save_path: Path | None = None,
        measured_save_path: Path | None = None,
        warn: bool = True,
        measurement_area_x_mm: float | None = None,
        measurement_area_y_mm: float | None = None,
    ):
        if not measured_path or not reference_path:
            raise ValueError("Both measured_path and reference_path are required.")

        self.noise_floor_wkg = float(noise_floor_wkg)
        self.resample_resolution = resample_resolution

        measured_df = self._read_csv(measured_path)
        reference_df = self._read_csv(reference_path)

        # V13/V19: filter measured data to declared measurement area, centred on the
        # midpoint of the imported data grid (mean of x/y extents), before mask
        # computation, registration, and gamma evaluation.
        if measurement_area_x_mm is not None and measurement_area_y_mm is not None:
            cx_m = (measured_df["x_m"].min() + measured_df["x_m"].max()) / 2.0
            cy_m = (measured_df["y_m"].min() + measured_df["y_m"].max()) / 2.0
            x_half_m = float(measurement_area_x_mm) / 2.0 / 1000.0
            y_half_m = float(measurement_area_y_mm) / 2.0 / 1000.0
            measured_df = measured_df[
                ((measured_df["x_m"] - cx_m).abs() <= x_half_m)
                & ((measured_df["y_m"] - cy_m).abs() <= y_half_m)
            ].copy()
            if len(measured_df) == 0:
                raise CsvFormatError(
                    f"No measured data falls within the declared measurement area "
                    f"({measurement_area_x_mm:.0f} × {measurement_area_y_mm:.0f} mm). "
                    "Check that the area contains the measurement scan region."
                )

        meas_grid = self._to_grid(measured_df, resample_resolution)
        ref_grid = self._to_grid(reference_df, resample_resolution)

        meas_sar = meas_grid.sar_grid
        meas_support = meas_grid.support_mask
        meas_axes = meas_grid.axes
        meas_spacing = meas_grid.spacing

        ref_sar = ref_grid.sar_grid
        ref_support = ref_grid.support_mask
        ref_axes = ref_grid.axes
        ref_spacing = ref_grid.spacing

        if warn:
            self._check_grids(measured_df, reference_df, meas_spacing, ref_spacing)

        threshold_cap_wkg = 0.1
        cutoff_wkg = float(min(threshold_cap_wkg, 2.0 * self.noise_floor_wkg))

        self.measured_raw_peak = float(np.max(meas_sar)) if meas_sar.size > 0 else 0.0
        meas_mask = (meas_sar >= cutoff_wkg) & meas_support
        ref_mask = (ref_sar >= cutoff_wkg) & ref_support

        meas_noise_floor_mask = meas_support & ~(meas_sar >= cutoff_wkg)
        ref_noise_floor_mask = ref_support & ~(ref_sar >= cutoff_wkg)

        self.measured_peak = (
            float(meas_sar[meas_mask].max()) if np.any(meas_mask) else 1e-12
        )
        self.reference_peak = (
            float(ref_sar[ref_mask].max()) if np.any(ref_mask) else 1e-12
        )
        self.measured_peak = max(self.measured_peak, 1e-12)
        self.reference_peak = max(self.reference_peak, 1e-12)
        self.measured_peak_30dbm = self.measured_peak * (
            10 ** ((30.0 - float(power_level_dbm)) / 10.0)
        )
        self.scaling_error = (self.measured_peak_30dbm / self.reference_peak) - 1.0

        meas_lin = np.where(meas_support, meas_sar / self.measured_peak, 0.0).astype(
            np.float32
        )
        ref_lin = np.where(ref_support, ref_sar / self.reference_peak, 0.0).astype(
            np.float32
        )

        self.measured_image_linear = self._array_to_sitk(
            meas_lin, *meas_spacing, meas_axes[0], meas_axes[1]
        )
        self.reference_image_linear = self._array_to_sitk(
            ref_lin, *ref_spacing, ref_axes[0], ref_axes[1]
        )

        meas_floor = cutoff_wkg / self.measured_peak
        ref_floor = cutoff_wkg / self.reference_peak
        shared_floor = float(max(meas_floor, ref_floor, 1e-12))

        self.measured_image_db = self._linear_to_db(
            self.measured_image_linear, shared_floor
        )
        self.reference_image_db = self._linear_to_db(
            self.reference_image_linear, shared_floor
        )

        self._measured_mask_abs = meas_mask
        self._reference_mask_abs = ref_mask
        self._measured_support_mask = meas_support
        self._reference_support_mask = ref_support
        self._measured_noise_floor_mask = meas_noise_floor_mask
        self._reference_noise_floor_mask = ref_noise_floor_mask
        self._measured_axes_m = meas_axes
        self._reference_axes_m = ref_axes

        if show_plot:
            self.plot(
                image_save_path=image_save_path,
                reference_save_path=reference_save_path,
                measured_save_path=measured_save_path,
            )

    @staticmethod
    def _is_recognizable_coordinate(header: str, coord: str) -> bool:
        """
        Check if a header is a recognizable coordinate column for x or y.

        Valid formats:
        - "x" or "y" (bare)
        - "x_m", "y_m" (meter underscore)
        - "x_mm", "y_mm" (millimeter underscore)
        - "x [m]", "y [m]" (meter bracket)
        - "x [mm]", "y [mm]" (millimeter bracket)

        Invalid: "xpos", "ypos", etc.
        """
        header_lower = header.lower()
        # Exact match for bare x or y
        if header_lower == coord:
            return True
        # Unit suffixes with underscore or bracket
        if header_lower in (
            f"{coord}_m",
            f"{coord}_mm",
            f"{coord} [m]",
            f"{coord} [mm]",
            f"{coord} [m]",
            f"{coord} [mm]",
        ):
            return True
        # More general: coord followed by underscore and a unit, or space-bracket pattern
        if header_lower.startswith(f"{coord}_"):
            suffix = header_lower[len(coord) + 1 :]
            return suffix in ("m", "mm")
        if header_lower.startswith(f"{coord} [") and header_lower.endswith("]"):
            unit = header_lower[len(coord) + 2 : -1].strip()
            return unit in ("m", "mm")
        return False

    @staticmethod
    def _read_csv(path: str) -> pd.DataFrame:
        """Return DataFrame with columns x_m, y_m (meters) and sar_wkg (W/kg)."""
        header_row = None
        try:
            with open(path, encoding="utf-8") as f:
                for i in range(50):
                    line = f.readline()
                    if not line:
                        break
                    if (
                        "x" in line.replace(" ", "").lower()
                        and "y" in line.replace(" ", "").lower()
                    ):
                        header_row = i
                        break
        except OSError as exc:
            raise CsvFormatError(f"Could not read CSV: {path}") from exc
        if header_row is None:
            raise CsvFormatError(f"No recognizable x/y coordinate columns in: {path}")

        try:
            df = pd.read_csv(path, header=header_row, skipinitialspace=True)
        except (pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
            raise CsvFormatError(f"Failed to parse CSV: {path}") from exc
        df.columns = [c.strip().lower() for c in df.columns]

        # Find recognizable x and y columns
        xcols = [
            c for c in df.columns if SARImageLoader._is_recognizable_coordinate(c, "x")
        ]
        ycols = [
            c for c in df.columns if SARImageLoader._is_recognizable_coordinate(c, "y")
        ]

        if not xcols or not ycols:
            raise CsvFormatError(
                f"No recognizable x/y coordinate columns in: {path}\nFound: {list(df.columns)}"
            )
        if len(xcols) > 1:
            raise CsvFormatError(
                f"Ambiguous x-coordinate column(s) in: {path}. Found: {xcols}"
            )
        if len(ycols) > 1:
            raise CsvFormatError(
                f"Ambiguous y-coordinate column(s) in: {path}. Found: {ycols}"
            )

        xcol = xcols[0]
        ycol = ycols[0]

        sarcol = (
            "sar"
            if "sar" in df.columns
            else next((c for c in df.columns if "sar" in c), None)
        )
        if not sarcol:
            raise CsvFormatError(
                f"Missing SAR column in: {path}\nFound: {list(df.columns)}"
            )
        assert xcol is not None
        assert ycol is not None
        assert sarcol is not None

        out = pd.DataFrame(
            {
                "x_m": pd.to_numeric(df[xcol], errors="coerce"),
                "y_m": pd.to_numeric(df[ycol], errors="coerce"),
                "sar_wkg": pd.to_numeric(df[sarcol], errors="coerce"),
            }
        ).dropna()

        scale = SARImageLoader._coordinate_scale_to_meters(xcol, ycol)
        out["x_m"] = out["x_m"].astype(float) * scale
        out["y_m"] = out["y_m"].astype(float) * scale
        out["sar_wkg"] = np.abs(out["sar_wkg"].astype(float))
        return out

    @staticmethod
    def _coordinate_unit(header: str) -> str:
        """
        Extract the unit from a coordinate header string.

        Examples:
            "x_mm" -> "mm"
            "x [mm]" -> "mm"
            "x_m" -> "m"
            "x [m]" -> "m"
            "x" -> "m" (default to meters)

        Returns:
            "mm" if millimeters are explicitly indicated, else "m"
        """
        header_lower = header.lower()

        # Check for explicit "mm" in the header
        if "mm" in header_lower:
            return "mm"

        # Check for explicit "m" in brackets (but not "mm")
        if "[m]" in header_lower:
            return "m"

        # Default to meters for bare headers like "x" or "y"
        return "m"

    @staticmethod
    def _coordinate_scale_to_meters(xcol: str, ycol: str) -> float:
        """Infer coordinate units from the headers and return the scale to meters."""

        def unit_scale(header: str) -> float:
            unit = SARImageLoader._coordinate_unit(header)
            if unit == "mm":
                return 1e-3
            return 1.0

        x_scale = unit_scale(xcol)
        y_scale = unit_scale(ycol)
        if not np.isclose(x_scale, y_scale):
            raise CsvFormatError(
                "Inconsistent coordinate units between x/y headers: "
                f"{xcol!r} and {ycol!r}."
            )
        return float(x_scale)

    @staticmethod
    def _to_grid(df: pd.DataFrame, step_m: float | None) -> Grid:
        """Returns Grid object with sar_grid, support_mask, axes, and spacing."""
        x = df["x_m"].to_numpy(dtype=float)
        y = df["y_m"].to_numpy(dtype=float)
        sar = df["sar_wkg"].to_numpy(dtype=float)

        if step_m is not None:
            step_m = float(step_m)
            x_axis = np.arange(x.min(), x.max() + 0.5 * step_m, step_m)
            y_axis = np.arange(y.min(), y.max() + 0.5 * step_m, step_m)
            X, Y = np.meshgrid(x_axis, y_axis, indexing="xy")

            sar_interp = griddata(
                np.column_stack((x, y)),
                sar,
                (X, Y),
                method="linear",
                fill_value=np.nan,
            )
            support = np.isfinite(sar_interp)
            sar_grid = np.where(support, sar_interp, 0.0)
            return Grid(
                sar_grid=sar_grid,
                support_mask=support,
                x_axis_m=x_axis,
                y_axis_m=y_axis,
                dx_m=step_m,
                dy_m=step_m,
            )

        x_axis = np.unique(np.sort(x))
        y_axis = np.unique(np.sort(y))

        sar_grid = (
            df.pivot_table(index="y_m", columns="x_m", values="sar_wkg", aggfunc="max")
            .reindex(index=y_axis, columns=x_axis, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        count_grid = (
            df.pivot_table(
                index="y_m", columns="x_m", values="sar_wkg", aggfunc="count"
            )
            .reindex(index=y_axis, columns=x_axis, fill_value=0)
            .to_numpy(dtype=int)
        )
        support = count_grid > 0

        dx = float(np.median(np.diff(x_axis))) if len(x_axis) > 1 else 1e-3
        dy = float(np.median(np.diff(y_axis))) if len(y_axis) > 1 else 1e-3
        return Grid(
            sar_grid=sar_grid,
            support_mask=support,
            x_axis_m=x_axis,
            y_axis_m=y_axis,
            dx_m=dx,
            dy_m=dy,
        )

    @staticmethod
    def _check_grids(meas_df, ref_df, meas_spacing, ref_spacing) -> None:
        """Check the completeness and consistency of measurement and reference grids."""

        def is_incomplete(df):
            nx = len(np.unique(df["x_m"].to_numpy(dtype=float)))
            ny = len(np.unique(df["y_m"].to_numpy(dtype=float)))
            return len(df) != nx * ny

        if is_incomplete(meas_df):
            warnings.warn(
                "Measured CSV is not a complete rectangular grid. "
                "Consider setting resample_resolution (e.g. 0.001).",
                UserWarning,
            )
        if is_incomplete(ref_df):
            warnings.warn(
                "Reference CSV is not a complete rectangular grid. "
                "Consider setting resample_resolution (e.g. 0.001).",
                UserWarning,
            )
        all_spacings = [*meas_spacing, *ref_spacing]
        if not all(np.isfinite(s) for s in all_spacings):
            warnings.warn("Non-finite spacing detected.", UserWarning)

    @staticmethod
    def _array_to_sitk(
        arr: np.ndarray,
        dx_m: float,
        dy_m: float,
        x_axis_m: np.ndarray,
        y_axis_m: np.ndarray,
    ) -> sitk.Image:
        img = sitk.GetImageFromArray(arr.astype(np.float32))
        img.SetSpacing([float(dx_m), float(dy_m)])
        img.SetOrigin([float(x_axis_m[0]), float(y_axis_m[0])])
        return img

    @staticmethod
    def _linear_to_db(img_linear: sitk.Image, floor_norm: float) -> sitk.Image:
        """Convert peak-normalized linear image to dB, clamping at floor_norm before log."""
        floor_norm = float(max(floor_norm, 1e-12))
        arr = sitk.GetArrayFromImage(img_linear).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr[arr < floor_norm] = floor_norm
        db = (10.0 * np.log10(arr)).astype(np.float32)
        out = sitk.GetImageFromArray(db)
        out.CopyInformation(img_linear)
        return out

    def get_images(self) -> tuple[sitk.Image, sitk.Image]:
        """Return (reference_image_db, measured_image_db)."""
        return self.reference_image_db, self.measured_image_db

    def make_metric_masks(self) -> tuple[sitk.Image, sitk.Image]:
        """Return (measured_mask, reference_mask) as uint8 SimpleITK images."""
        meas_mask = sitk.GetImageFromArray(self._measured_mask_abs.astype(np.uint8))
        ref_mask = sitk.GetImageFromArray(self._reference_mask_abs.astype(np.uint8))
        meas_mask.CopyInformation(self.measured_image_linear)
        ref_mask.CopyInformation(self.reference_image_linear)
        return meas_mask, ref_mask

    def make_support_masks(self) -> tuple[sitk.Image, sitk.Image]:
        """Return (measured_support, reference_support) as uint8 SimpleITK images."""
        meas_support = sitk.GetImageFromArray(
            self._measured_support_mask.astype(np.uint8)
        )
        ref_support = sitk.GetImageFromArray(
            self._reference_support_mask.astype(np.uint8)
        )
        meas_support.CopyInformation(self.measured_image_linear)
        ref_support.CopyInformation(self.reference_image_linear)
        return meas_support, ref_support

    def plot(
        self,
        image_save_path: Path | None = None,
        reference_save_path: Path | None = None,
        measured_save_path: Path | None = None,
        plotting_config: PlottingConfig | None = None,
    ) -> None:
        meas_arr = sitk.GetArrayFromImage(self.measured_image_linear).astype(float)
        ref_arr = sitk.GetArrayFromImage(self.reference_image_linear).astype(float)
        meas_unit = meas_arr / max(float(np.nanmax(meas_arr)), 1e-12)
        ref_unit = ref_arr / max(float(np.nanmax(ref_arr)), 1e-12)

        x_meas, y_meas = self._measured_axes_m
        x_ref, y_ref = self._reference_axes_m

        if image_save_path is not None:
            plot_loaded_images(
                measured_image=meas_unit,
                reference_image=ref_unit,
                measured_axes=(x_meas, y_meas),
                reference_axes=(x_ref, y_ref),
                measured_support_mask=self._measured_support_mask,
                reference_support_mask=self._reference_support_mask,
                measured_noise_floor_mask=self._measured_noise_floor_mask,
                reference_noise_floor_mask=self._reference_noise_floor_mask,
                image_save_path=image_save_path,
                plotting_config=plotting_config,
            )

        if (
            reference_save_path is None
            and measured_save_path is None
            and image_save_path is not None
        ):
            return

        plot_sar_image(
            sar_image=meas_unit,
            x_axis_m=x_meas,
            y_axis_m=y_meas,
            title="Measured, Normalized",
            xlabel="$x_e$ (mm)",
            ylabel="$y_e$ (mm)",
            save_path=measured_save_path,
            show_colorbar=False,
            support_mask=self._measured_support_mask,
            noise_floor_mask=self._measured_noise_floor_mask,
            plotting_config=plotting_config,
        )

        reference_plotting_config = plotting_config
        if plotting_config is not None:
            half = max(
                (plotting_config.window_mm[1] - plotting_config.window_mm[0]) / 2.0,
                (plotting_config.window_mm[3] - plotting_config.window_mm[2]) / 2.0,
            )
            reference_plotting_config = replace(
                plotting_config,
                center_x_mm=0.0,
                center_y_mm=0.0,
                window_mm=(-half, half, -half, half),
            )

        plot_sar_image(
            sar_image=ref_unit,
            x_axis_m=x_ref,
            y_axis_m=y_ref,
            title="Reference, Normalized",
            xlabel="$x_r$ (mm)",
            ylabel="$y_r$ (mm)",
            save_path=reference_save_path,
            show_colorbar=False,
            support_mask=self._reference_support_mask,
            noise_floor_mask=self._reference_noise_floor_mask,
            plotting_config=reference_plotting_config,
        )

    def plot_aligned(
        self,
        aligned_image: sitk.Image,
        aligned_meas_save_path: Path | None = None,
        plotting_config: PlottingConfig | None = None,
    ) -> None:
        aligned_arr = sitk.GetArrayFromImage(aligned_image).astype(float)
        arr_min = float(np.nanmin(aligned_arr))
        arr_max = float(np.nanmax(aligned_arr))
        aligned_unit = (aligned_arr - arr_min) / max(arr_max - arr_min, 1e-12)
        x_meas, y_meas = self._measured_axes_m
        plot_sar_image(
            sar_image=aligned_unit,
            x_axis_m=x_meas,
            y_axis_m=y_meas,
            title="Reference, After Registration",
            xlabel="$x'_r$ (mm)",
            ylabel="$y'_r$ (mm)",
            save_path=aligned_meas_save_path,
            show_colorbar=True,
            support_mask=self._measured_support_mask,
            noise_floor_mask=self._measured_noise_floor_mask,
            plotting_config=plotting_config,
        )
