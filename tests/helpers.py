from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _coord_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = [str(c).strip().lower() for c in df.columns]
    mapping = dict(zip(cols, df.columns, strict=False))
    xcol = next((mapping[c] for c in cols if c == "x" or c.startswith("x")), None)
    ycol = next((mapping[c] for c in cols if c == "y" or c.startswith("y")), None)
    if xcol is None or ycol is None:
        raise KeyError("Could not identify x/y columns in dataframe.")
    return str(xcol), str(ycol)


def gaussian_2d(x, y, x0=0.0, y0=0.0, sx=0.02, sy=0.03, peak=1.0):
    """
    Build a 2D Gaussian on a rectilinear mesh.

    Args:
        x, y: 1D arrays defining the grid (meters).
        x0, y0: Gaussian center (meters).
        sx, sy: Std-dev along x/y (meters).
        peak: Peak amplitude (linear units).

    Returns:
        (X, Y, G): meshgrid coordinates and Gaussian values with shape (len(y), len(x)).
    """
    X, Y = np.meshgrid(x, y, indexing="xy")
    G = peak * np.exp(-((X - x0) ** 2 / (2 * sx**2) + (Y - y0) ** 2 / (2 * sy**2)))
    return X, Y, G


def make_rect_grid(xmin=-0.1, xmax=0.1, ymin=-0.1, ymax=0.1, step=0.005):
    """
    Build 1D coordinate axes for a rectilinear grid.

    Args:
        xmin, xmax, ymin, ymax: extents (meters).
        step: lattice spacing (meters).

    Returns:
        (x, y): 1D arrays.
    """
    x = np.arange(xmin, xmax + 0.5 * step, step, dtype=float)
    y = np.arange(ymin, ymax + 0.5 * step, step, dtype=float)
    return x, y


def write_sar_csv(
    path,
    x,
    y,
    Z,
    z_level=0.0,
    noise_std=0.0,
    coordinate_headers: tuple[str, str] = ("x [m]", "y [m]"),
):
    """
    Write a tidy SAR CSV (columns: x, y, z, sar).

    Args:
        path: file path to write.
        x, y: 1D axes.
        Z: 2D values with shape (len(y), len(x)).
        z_level: constant z column.
        noise_std: optional Gaussian noise on values.
        coordinate_headers: names to use for the x/y columns.

    Returns:
        None (writes CSV).
    """
    rng = np.random.default_rng(123)
    vals = Z.T.reshape(-1)
    if noise_std > 0:
        vals = vals + rng.normal(0.0, noise_std, Z.size)
    df = pd.DataFrame(
        {
            coordinate_headers[0]: np.repeat(x, len(y)),
            coordinate_headers[1]: np.tile(y, len(x)),
            "z": z_level,
            "sar": np.clip(vals, a_min=0.0, a_max=None),
        }
    )
    df.to_csv(path, index=False)


def punch_circular_hole(df, center=(0.03, -0.02), radius=0.03):
    """
    Remove rows whose (x,y) are inside a circle.

    Args:
        df: tidy SAR dataframe (x,y,z,sar).
        center: (cx, cy) meters.
        radius: meters.

    Returns:
        Filtered dataframe (copy).
    """
    cx, cy = map(float, center)
    xcol, ycol = _coord_columns(df)
    m = ((df[xcol] - cx) ** 2 + (df[ycol] - cy) ** 2) >= radius**2
    return df.loc[m].reset_index(drop=True)


def punch_rect_hole(df, xmin, xmax, ymin, ymax):
    """
    Remove rows whose (x,y) fall in [xmin,xmax]×[ymin,ymax].

    Returns:
        Filtered dataframe (copy).
    """
    xcol, ycol = _coord_columns(df)
    m = ~(
        (df[xcol] >= xmin)
        & (df[xcol] <= xmax)
        & (df[ycol] >= ymin)
        & (df[ycol] <= ymax)
    )
    return df.loc[m].reset_index(drop=True)


def rigid_transform_points(df, tx=0.0, ty=0.0, theta_deg=0.0):
    """
    Apply a rigid transform to (x,y) points.

    Args:
        df: tidy SAR dataframe.
        tx, ty: translation (meters).
        theta_deg: rotation angle (degrees), CCW.

    Returns:
        Copy of df with transformed x,y.
    """
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    xcol, ycol = _coord_columns(df)
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    out = df.copy()
    out[xcol] = c * x - s * y + tx
    out[ycol] = s * x + c * y + ty
    return out


def compare_gamma_maps(
    expected_gamma: np.ndarray,
    expected_mask: np.ndarray,
    actual_gamma: np.ndarray,
    actual_mask: np.ndarray,
    threshold: float | None = None,
):
    """
    Compare two gamma maps for near-equality.

    Args:
        expected_gamma, expected_mask: 2D arrays of expected gamma values and evaluation mask.
        actual_gamma, actual_mask: 2D arrays of actual gamma values and evaluation mask.
        threshold: optional absolute tolerance for gamma value differences (otherwise read environment variable).
    """

    assert actual_gamma.shape == expected_gamma.shape
    assert actual_mask.shape == expected_mask.shape
    assert np.array_equal(actual_mask, expected_mask)

    finite = np.isfinite(expected_gamma) & np.isfinite(actual_gamma)
    assert np.array_equal(np.isfinite(expected_gamma), np.isfinite(actual_gamma))

    if threshold is None:
        threshold = float(os.getenv("MEASUREMENT_GAMMA_DIFF_THRESHOLD", "2e-6"))

    assert np.allclose(
        expected_gamma[finite],
        actual_gamma[finite],
        atol=threshold,
    )
