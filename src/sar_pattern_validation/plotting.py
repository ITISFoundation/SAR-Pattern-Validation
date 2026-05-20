from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from sar_pattern_validation.utils import image_extent_mm as _shared_image_extent_mm
from sar_pattern_validation.workflow_config import PlottingConfig


def image_extent_mm(img: sitk.Image) -> tuple[float, float, float, float]:
    extent = _shared_image_extent_mm(img)
    return (extent[0], extent[1], extent[2], extent[3])


def extent_mm_from_axes(
    x_axis_m: np.ndarray, y_axis_m: np.ndarray
) -> tuple[float, float, float, float]:
    dx = float(np.median(np.diff(x_axis_m))) if len(x_axis_m) > 1 else 0.0
    dy = float(np.median(np.diff(y_axis_m))) if len(y_axis_m) > 1 else 0.0
    return (
        1000 * (x_axis_m.min() - 0.5 * dx),
        1000 * (x_axis_m.max() + 0.5 * dx),
        1000 * (y_axis_m.min() - 0.5 * dy),
        1000 * (y_axis_m.max() + 0.5 * dy),
    )


def show_registration_overlay(
    fixed_image: sitk.Image,
    aligned_moving_image: sitk.Image,
    *,
    title: str = "Rigid Registration Overlay",
    image_save_path: Path | None = None,
    noise_floor_mask: np.ndarray | None = None,
    plotting_config: PlottingConfig | None = None,
) -> None:
    config = plotting_config or PlottingConfig()
    fixed_u8 = sitk.GetArrayFromImage(sitk.RescaleIntensity(fixed_image)).astype(
        np.uint8
    )
    aligned_u8 = sitk.GetArrayFromImage(
        sitk.RescaleIntensity(aligned_moving_image)
    ).astype(np.uint8)
    if fixed_u8.shape != aligned_u8.shape:
        raise ValueError("Fixed and aligned images must have the same shape.")

    overlay = np.zeros((*fixed_u8.shape, 3), dtype=np.uint8)
    overlay[..., 0] = fixed_u8
    overlay[..., 2] = aligned_u8

    fixed_extent = image_extent_mm(fixed_image)

    with _style_context(config):
        fig, ax = plt.subplots(figsize=config.single_figure_size)
        fig.set_facecolor(config.figure_facecolor)
        ax.set_facecolor(config.dark_axes_facecolor)
        ax.tick_params(axis="both", which="major")
        ax.imshow(
            overlay,
            extent=fixed_extent,
            origin="lower",
            aspect="equal",
        )
        ax.set_title(_two_line_title(title))
        ax.set(
            xlim=(config.window_mm[0], config.window_mm[1]),
            ylim=(config.window_mm[2], config.window_mm[3]),
        )
        ax.set_xlabel("$x'_r$ (mm)")
        ax.set_ylabel("$y'_r$ (mm)")

        noise_floor_handle = _overlay_noise_floor(
            ax,
            noise_floor_mask=noise_floor_mask,
            extent_mm=fixed_extent,
            config=config,
        )

        patches = [
            mpatches.Patch(color="red", label="Measured"),
            mpatches.Patch(color="blue", label="Reference"),
            mpatches.Patch(color="magenta", label="Overlap"),
        ]
        if noise_floor_handle is not None:
            patches.append(noise_floor_handle)
        legend = ax.legend(handles=patches, loc="lower right", frameon=True)
        legend.get_frame().set_facecolor(config.dark_axes_facecolor)
        legend.get_frame().set_edgecolor("w")
        for text in legend.get_texts():
            text.set_color("w")

        fig.tight_layout()
        _save_or_show(fig, image_save_path, config)


def plot_loaded_images(
    *,
    measured_image: np.ndarray,
    reference_image: np.ndarray,
    measured_axes: tuple[np.ndarray, np.ndarray],
    reference_axes: tuple[np.ndarray, np.ndarray],
    measured_support_mask: np.ndarray | None = None,
    reference_support_mask: np.ndarray | None = None,
    measured_noise_floor_mask: np.ndarray | None = None,
    reference_noise_floor_mask: np.ndarray | None = None,
    image_save_path: Path | None = None,
    plotting_config: PlottingConfig | None = None,
) -> None:
    config = plotting_config or PlottingConfig()

    with _style_context(config):
        fig, (ax0, ax1) = plt.subplots(
            1,
            2,
            figsize=config.combined_figure_size,
            facecolor=config.figure_facecolor,
        )
        common = dict(origin="lower", aspect="equal", cmap="hot")

        sar_im = None
        for ax, data, axes, support_mask, noise_floor_mask, label in [
            (
                ax0,
                measured_image,
                measured_axes,
                measured_support_mask,
                measured_noise_floor_mask,
                "Measured",
            ),
            (
                ax1,
                reference_image,
                reference_axes,
                reference_support_mask,
                reference_noise_floor_mask,
                "Reference",
            ),
        ]:
            extent = extent_mm_from_axes(*axes)
            im = ax.imshow(data, extent=extent, **common)
            ax.set_title(_two_line_title(f"{label} SAR (linear, unit-peak)"))
            ax.set_facecolor(config.not_evaluated_color)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            noise_floor_handle = _overlay_noise_floor(
                ax,
                noise_floor_mask=noise_floor_mask,
                extent_mm=extent,
                config=config,
            )
            cropped_handle = _overlay_cropped_measurement_data(
                ax,
                axes=axes,
                support_mask=support_mask,
                config=config,
            )
            _apply_overlay_legend(ax, [noise_floor_handle, cropped_handle], config)
            if sar_im is None:
                sar_im = im

        if image_save_path is not None and sar_im is not None and config.save_colorbars:
            colorbar_path = _derive_colorbar_path(image_save_path, "sar_colorbar")
            _save_colorbar_only(
                sar_im,
                colorbar_path,
                config,
                label="Normalized SAR",
                orientation="vertical",
            )

        fig.tight_layout()
        _save_or_show(fig, image_save_path, config)


def plot_sar_image(
    *,
    sar_image: np.ndarray,
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path | None = None,
    show_colorbar: bool,
    support_mask: np.ndarray | None = None,
    noise_floor_mask: np.ndarray | None = None,
    plotting_config: PlottingConfig | None = None,
) -> None:
    config = plotting_config or PlottingConfig()

    with _style_context(config):
        fig, ax = plt.subplots(figsize=config.single_figure_size)
        fig.set_facecolor(config.figure_facecolor)
        ax.set_facecolor(config.not_evaluated_color)
        extent = extent_mm_from_axes(x_axis_m, y_axis_m)
        im = ax.imshow(
            sar_image,
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            extent=extent,
            origin="lower",
            aspect="equal",
        )
        ax.set_title(_two_line_title(title))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(config.window_mm[0], config.window_mm[1])
        ax.set_ylim(config.window_mm[2], config.window_mm[3])
        noise_floor_handle = _overlay_noise_floor(
            ax,
            noise_floor_mask=noise_floor_mask,
            extent_mm=extent,
            config=config,
        )
        cropped_handle = _overlay_cropped_measurement_data(
            ax,
            axes=(x_axis_m, y_axis_m),
            support_mask=support_mask,
            config=config,
        )
        _apply_overlay_legend(ax, [noise_floor_handle, cropped_handle], config)

        if show_colorbar and save_path is not None and config.save_colorbars:
            colorbar_path = _derive_colorbar_path(save_path, "colorbar")
            _save_colorbar_only(
                im,
                colorbar_path,
                config,
                label="Normalized SAR",
                orientation="vertical",
            )
        fig.tight_layout()

        _save_or_show(fig, save_path, config)


def _overlay_measurement_limit_mask(ax, config: PlottingConfig) -> None:
    x_mm = config.measurement_area_x_mm
    y_mm = config.measurement_area_y_mm
    if x_mm is None or y_mm is None:
        return
    x_half = x_mm / 2.0
    y_half = y_mm / 2.0
    side_half = max(x_half, y_half)
    cx = config.center_x_mm
    cy = config.center_y_mm
    grey = config.not_evaluated_color
    if y_half < side_half:
        strip_h = side_half - y_half
        for y0 in (cy + y_half, cy - side_half):
            ax.add_patch(
                mpatches.Rectangle(
                    (cx - side_half, y0),
                    2 * side_half,
                    strip_h,
                    color=grey,
                    transform=ax.transData,
                    zorder=5,
                    linewidth=0,
                )
            )
    if x_half < side_half:
        strip_w = side_half - x_half
        for x0 in (cx - side_half, cx + x_half):
            ax.add_patch(
                mpatches.Rectangle(
                    (x0, cy - side_half),
                    strip_w,
                    2 * side_half,
                    color=grey,
                    transform=ax.transData,
                    zorder=5,
                    linewidth=0,
                )
            )


def _compute_cropped_data_mask(
    *,
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    support_mask: np.ndarray,
    config: PlottingConfig,
) -> np.ndarray | None:
    x_mm = config.measurement_area_x_mm
    y_mm = config.measurement_area_y_mm
    if x_mm is None or y_mm is None:
        return None

    if support_mask.shape != (len(y_axis_m), len(x_axis_m)):
        return None

    cx = float(config.center_x_mm)
    cy = float(config.center_y_mm)
    x_half = float(x_mm) / 2.0
    y_half = float(y_mm) / 2.0
    x_axis_mm = x_axis_m * 1000.0
    y_axis_mm = y_axis_m * 1000.0

    outside_x = (x_axis_mm < (cx - x_half)) | (x_axis_mm > (cx + x_half))
    outside_y = (y_axis_mm < (cy - y_half)) | (y_axis_mm > (cy + y_half))
    outside_rect = outside_y[:, None] | outside_x[None, :]
    return support_mask & outside_rect


def _overlay_cropped_measurement_data(
    ax,
    *,
    axes: tuple[np.ndarray, np.ndarray],
    support_mask: np.ndarray | None,
    config: PlottingConfig,
) -> mpatches.Patch | None:
    if support_mask is None:
        return None

    x_axis_m, y_axis_m = axes
    cropped_mask = _compute_cropped_data_mask(
        x_axis_m=x_axis_m,
        y_axis_m=y_axis_m,
        support_mask=support_mask,
        config=config,
    )
    if cropped_mask is None or not np.any(cropped_mask):
        return None

    cropped_overlay = np.ma.masked_where(~cropped_mask, cropped_mask.astype(float))
    cmap = mpl.colors.ListedColormap([config.cropped_data_color])
    ax.imshow(
        cropped_overlay,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=extent_mm_from_axes(x_axis_m, y_axis_m),
        origin="lower",
        aspect="equal",
        alpha=0.6,
        zorder=6,
    )
    return mpatches.Patch(
        color=config.cropped_data_color,
        alpha=0.6,
        label="Cropped (outside meas. area)",
    )


def _overlay_noise_floor(
    ax,
    *,
    noise_floor_mask: np.ndarray | None,
    extent_mm: tuple[float, float, float, float],
    config: PlottingConfig,
) -> mpatches.Patch | None:
    if noise_floor_mask is None or not np.any(noise_floor_mask):
        return None

    overlay = np.ma.masked_where(~noise_floor_mask, noise_floor_mask.astype(float))
    cmap = mpl.colors.ListedColormap([config.noise_floor_color])
    ax.imshow(
        overlay,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=extent_mm,
        origin="lower",
        aspect="equal",
        alpha=0.45,
        zorder=4,
    )
    return mpatches.Patch(
        color=config.noise_floor_color,
        alpha=0.45,
        label="Noise",
    )


def _apply_overlay_legend(
    ax,
    handles: list[mpatches.Patch | None],
    config: PlottingConfig,
    *,
    on_dark_axes: bool = False,
) -> None:
    filtered = [h for h in handles if h is not None]
    if not filtered:
        return
    legend = ax.legend(
        handles=filtered,
        loc="lower right",
        frameon=True,
        fontsize=7,
        framealpha=0.0,
    )
    if on_dark_axes:
        legend.get_frame().set_facecolor(config.dark_axes_facecolor)
        legend.get_frame().set_edgecolor("w")
        for text in legend.get_texts():
            text.set_color("w")


def plot_gamma_results(
    *,
    gamma_map: np.ndarray,
    evaluation_mask: np.ndarray,
    gamma_cap: float,
    extent_mm: tuple[float, float, float, float],
    gamma_image_save_path: Path | None = None,
    failure_image_save_path: Path | None = None,
    noise_floor_mask: np.ndarray | None = None,
    plotting_config: PlottingConfig | None = None,
) -> None:
    config = plotting_config or PlottingConfig()
    masked_gamma = np.ma.masked_invalid(gamma_map)
    failures = (gamma_map > 1.0) & evaluation_mask

    with _style_context(config):
        fig, ax = plt.subplots(figsize=config.single_figure_size)
        fig.set_facecolor(config.figure_facecolor)
        ax.set_facecolor(config.not_evaluated_color)

        cmap = mpl.colormaps["plasma"].copy()
        cmap.set_bad(config.not_evaluated_color)

        im = ax.imshow(
            masked_gamma,
            cmap=cmap,
            vmin=0,
            vmax=gamma_cap,
            interpolation="nearest",
            extent=extent_mm,
            origin="lower",
            aspect="equal",
        )
        ax.set_title(_two_line_title("Gamma Index"))
        ax.set_xlim(config.window_mm[0], config.window_mm[1])
        ax.set_ylim(config.window_mm[2], config.window_mm[3])
        ax.set_xlabel("$x'_r$ (mm)")
        ax.set_ylabel("$y'_r$ (mm)")
        _overlay_measurement_limit_mask(ax, config)
        noise_floor_handle = _overlay_noise_floor(
            ax,
            noise_floor_mask=noise_floor_mask,
            extent_mm=extent_mm,
            config=config,
        )
        _apply_overlay_legend(ax, [noise_floor_handle], config)
        fig.tight_layout()
        if gamma_image_save_path is not None and config.save_colorbars:
            colorbar_path_v = _derive_colorbar_path(
                gamma_image_save_path, "colorbar_vertical"
            )
            colorbar_path_h = _derive_colorbar_path(
                gamma_image_save_path, "colorbar_horizontal"
            )
            _save_colorbar_only(
                im,
                colorbar_path_v,
                config,
                label="Gamma Index",
                orientation="vertical",
            )
            _save_colorbar_only(
                im,
                colorbar_path_h,
                config,
                label="Gamma Index",
                orientation="horizontal",
            )
        _save_or_show(fig, gamma_image_save_path, config)

    with _style_context(config):
        fig, ax = plt.subplots(figsize=config.single_figure_size)
        fig.set_facecolor(config.figure_facecolor)
        ax.set_facecolor(config.not_evaluated_color)
        eval_masked = np.ma.masked_where(
            evaluation_mask == 0, evaluation_mask.astype(float)
        )
        cmap_eval = mpl.colormaps["gray"].copy()
        cmap_eval.set_bad(config.not_evaluated_color)
        ax.imshow(
            eval_masked,
            cmap=cmap_eval,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            extent=extent_mm,
            origin="lower",
            aspect="equal",
        )
        ax.imshow(
            failures.astype(float),
            cmap="Reds",
            interpolation="nearest",
            alpha=0.85,
            extent=extent_mm,
            origin="lower",
            aspect="equal",
        )
        ax.set_title(_two_line_title("Gamma Pass / Fail Map"))
        ax.set_xlim(config.window_mm[0], config.window_mm[1])
        ax.set_ylim(config.window_mm[2], config.window_mm[3])
        ax.set_xlabel("$x'_r$ (mm)")
        ax.set_ylabel("$y'_r$ (mm)")
        _overlay_measurement_limit_mask(ax, config)
        noise_floor_handle = _overlay_noise_floor(
            ax,
            noise_floor_mask=noise_floor_mask,
            extent_mm=extent_mm,
            config=config,
        )
        handles = [
            mpatches.Patch(facecolor="white", edgecolor="k", label="Pass"),
            mpatches.Patch(
                facecolor=mpl.colormaps["Reds"](0.8), edgecolor="k", label="Fail"
            ),
        ]
        if noise_floor_handle is not None:
            handles.append(noise_floor_handle)
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.0)
        fig.tight_layout()
        _save_or_show(fig, failure_image_save_path, config)


def _style_context(config: PlottingConfig):
    return mpl.rc_context({"font.size": config.font_size})


def _add_matched_colorbar(
    im,
    ax,
    config: PlottingConfig,
    *,
    label: str,
    width: float = 0.03,
    pad: float = 0.01,
):
    fig = ax.figure
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
    return fig.colorbar(im, cax=cax, label=label)


def _save_or_show(fig: Figure, save_path: Path | None, config: PlottingConfig) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=config.save_dpi)
        plt.close(fig)
        return
    plt.show()


def _derive_colorbar_path(save_path: Path, suffix: str) -> Path:
    return save_path.with_name(f"{save_path.stem}_{suffix}{save_path.suffix}")


def _save_colorbar_only(
    im,
    save_path: Path,
    config: PlottingConfig,
    *,
    label: str,
    orientation: str = "vertical",
) -> None:
    if orientation == "vertical":
        figsize = (0.8, 4.0)
        cax_bounds: tuple[float, float, float, float] = (0.15, 0.1, 0.1, 0.8)
    else:
        figsize = (4.0, 0.8)
        cax_bounds = (0.1, 0.15, 0.8, 0.1)

    fig = plt.figure(figsize=figsize)
    fig.set_facecolor(config.figure_facecolor)
    cax = fig.add_axes(cax_bounds)
    sm = ScalarMappable(norm=im.norm, cmap=im.cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation=orientation)
    cbar.set_label(label)
    fig.savefig(save_path, dpi=config.save_dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _two_line_title(title: str, max_chars_per_line: int = 32) -> str:
    if len(title) <= max_chars_per_line:
        return title

    wrapped = textwrap.wrap(title, width=max_chars_per_line)
    if len(wrapped) <= 2:
        return "\n".join(wrapped)

    return wrapped[0] + "\n" + " ".join(wrapped[1:])
