"""Savitzky-Golay Filter (Reference Sec 47.142).

Savitzky & Golay 1964 'Smoothing and differentiation of data by
simplified least-squares procedures', Anal Chem 36. Sliding-window
polynomial least-squares smoother:

    within window of length w=2m+1, fit degree-p polynomial by OLS;
    replace center value with the fitted polynomial's value at 0.

Equivalent to FIR convolution with a fixed coefficient vector that
depends only on (w, p, derivative-order). Preserves higher moments
of the signal better than moving average.
"""
from __future__ import annotations    # stdlib

import math    # factorial for derivative scaling

import numpy as np    # numerical arrays


def sg_coeffs(window, poly_order, deriv=0):
    """Compute Savitzky-Golay FIR coefficients for smoothing / differentiation."""
    if window % 2 == 0 or window < 1:
        raise ValueError("window must be odd and positive")
    half = window // 2
    x = np.arange(-half, half + 1)
    # Vandermonde matrix of powers 0..poly_order
    A = np.vander(x, poly_order + 1, increasing=True)
    # Pseudo-inverse row for derivative 'deriv' at x=0, scaled by deriv!
    pinv = np.linalg.pinv(A)
    coeffs = pinv[deriv] * float(math.factorial(deriv))
    return coeffs


def sg_smooth(y, window=11, poly_order=3, deriv=0):
    """Apply SG filter to 1-D signal with edge-mirror padding."""
    y = np.asarray(y, dtype=float)
    c = sg_coeffs(window, poly_order, deriv)
    half = window // 2
    y_pad = np.concatenate([y[half:0:-1], y, y[-2:-half - 2:-1]])
    return np.convolve(y_pad, c[::-1], mode="valid")


if __name__ == "__main__":
    print("=== Savitzky-Golay filter (Savitzky-Golay 1964) ===\n")
    rng = np.random.default_rng(0)

    # Clean signal + noise
    t = np.linspace(0, 4 * np.pi, 400)
    clean = np.sin(t) + 0.3 * np.sin(3 * t)
    noise = rng.normal(scale=0.3, size=t.shape)
    y = clean + noise

    for w, p in [(11, 3), (21, 3), (51, 3), (21, 5)]:
        y_smooth = sg_smooth(y, window=w, poly_order=p)
        rmse = float(np.sqrt(np.mean((y_smooth - clean) ** 2)))
        print(f"  window = {w:2d}   poly = {p}   RMSE (vs clean) = {rmse:.4f}")

    print(f"\n  Raw-signal RMSE (no filter): {float(np.sqrt(np.mean((y - clean) ** 2))):.4f}")

    # Derivative estimation
    dy_true = np.cos(t) + 0.9 * np.cos(3 * t)
    dy_est = sg_smooth(y, window=31, poly_order=4, deriv=1) / (t[1] - t[0])
    d_rmse = float(np.sqrt(np.mean((dy_est - dy_true) ** 2)))
    print(f"\n  1st-derivative estimation via SG (w=31, p=4): RMSE = {d_rmse:.4f}")

    print("\n--- library cross-check (scipy.signal.savgol_filter; signal R / prospectr R) ---")
