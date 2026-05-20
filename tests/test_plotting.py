"""Tests for plotting fixes: axis labels (#7) and Pass legend color (#6)."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from sar_pattern_validation.image_loader import SARImageLoader
from sar_pattern_validation.plotting import (
    plot_gamma_results,
    show_registration_overlay,
)


def _tiny_sitk(rows: int = 10, cols: int = 10, spacing: float = 0.005) -> sitk.Image:
    arr = np.ones((rows, cols), dtype=np.float32) * 0.5
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((spacing, spacing))
    return img


@pytest.fixture
def capture_fig(monkeypatch):
    """Replace _save_or_show to capture figures instead of saving/showing them."""
    import matplotlib.pyplot as plt

    import sar_pattern_validation.plotting as _mod

    captured: list = []

    def _fake(fig, path, config):
        captured.append(fig)
        plt.close(fig)

    monkeypatch.setattr(_mod, "_save_or_show", _fake)
    return captured


# ---------------------------------------------------------------------------
# T14 — axis labels renamed x_e/y_e → x'_r/y'_r
# ---------------------------------------------------------------------------


def test_registration_overlay_uses_registered_frame_labels(capture_fig) -> None:
    img = _tiny_sitk()
    show_registration_overlay(fixed_image=img, aligned_moving_image=img)
    assert capture_fig, "no figure captured"
    ax = capture_fig[0].axes[0]
    assert ax.get_xlabel() == "$x'_r$ (mm)"
    assert ax.get_ylabel() == "$y'_r$ (mm)"


def test_gamma_index_uses_registered_frame_labels(capture_fig) -> None:
    n = 10
    gamma = np.full((n, n), 0.5)
    mask = np.ones((n, n), dtype=bool)
    plot_gamma_results(
        gamma_map=gamma,
        evaluation_mask=mask,
        gamma_cap=1.5,
        extent_mm=(-25.0, 25.0, -25.0, 25.0),
    )
    gamma_fig = capture_fig[0]
    ax = gamma_fig.axes[0]
    assert ax.get_xlabel() == "$x'_r$ (mm)"
    assert ax.get_ylabel() == "$y'_r$ (mm)"


def test_gamma_pass_fail_uses_registered_frame_labels(capture_fig) -> None:
    n = 10
    gamma = np.full((n, n), 0.5)
    mask = np.ones((n, n), dtype=bool)
    plot_gamma_results(
        gamma_map=gamma,
        evaluation_mask=mask,
        gamma_cap=1.5,
        extent_mm=(-25.0, 25.0, -25.0, 25.0),
    )
    pass_fail_fig = capture_fig[1]
    ax = pass_fail_fig.axes[0]
    assert ax.get_xlabel() == "$x'_r$ (mm)"
    assert ax.get_ylabel() == "$y'_r$ (mm)"


def test_plot_aligned_uses_registered_frame_labels(tmp_csv_pair, capture_fig) -> None:
    measured, reference = tmp_csv_pair
    loader = SARImageLoader(measured, reference, show_plot=False)
    loader.plot_aligned(loader.reference_image_linear)
    assert capture_fig, "no figure captured"
    ax = capture_fig[0].axes[0]
    assert ax.get_xlabel() == "$x'_r$ (mm)"
    assert ax.get_ylabel() == "$y'_r$ (mm)"


# ---------------------------------------------------------------------------
# T13 — Pass legend color matches actual pass-region white
# ---------------------------------------------------------------------------


def test_gamma_pass_fail_legend_pass_color_is_white(capture_fig) -> None:
    n = 10
    gamma = np.full((n, n), 0.5)
    mask = np.ones((n, n), dtype=bool)
    plot_gamma_results(
        gamma_map=gamma,
        evaluation_mask=mask,
        gamma_cap=1.5,
        extent_mm=(-25.0, 25.0, -25.0, 25.0),
    )
    pass_fail_fig = capture_fig[1]
    legend = pass_fail_fig.axes[0].get_legend()
    assert legend is not None, "no legend on pass/fail figure"
    pass_patch = next(
        (h for h in legend.legend_handles if h.get_label() == "Pass"), None
    )
    assert pass_patch is not None, "'Pass' handle not found in legend"
    rgba = pass_patch.get_facecolor()
    np.testing.assert_allclose(
        rgba[:3],
        [1.0, 1.0, 1.0],
        atol=1e-3,
        err_msg="Pass legend color must be white to match pass-region fill",
    )
