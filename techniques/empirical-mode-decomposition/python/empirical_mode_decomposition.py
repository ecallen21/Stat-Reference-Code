"""Empirical Mode Decomposition (EMD) (Reference Sec 47.93).

Huang, Shen, Long, Wu, Shih, Zheng, Yen, Tung & Liu 1998 'The
empirical mode decomposition and the Hilbert spectrum for
nonlinear and non-stationary time series analysis', Proc R Soc A.

Decompose a signal x(t) into Intrinsic Mode Functions (IMFs) via
SIFTING:

    1. Find local extrema of x.
    2. Fit upper (max) and lower (min) envelopes with cubic splines.
    3. m(t) = mean of envelopes.
    4. h(t) = x(t) - m(t).
    5. Repeat until h satisfies IMF conditions (equal #zero-
       crossings and #extrema, zero envelope mean).

Residue r(t) = x(t) - IMF. Iterate on r for next IMF. Data-driven,
no basis assumption, unlike Fourier / wavelet.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.interpolate import CubicSpline    # spline envelopes


def _extrema(x):
    diff = np.diff(x)
    maxima = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
    minima = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0] + 1
    return maxima, minima


def _sift_once(t, x):
    maxima, minima = _extrema(x)
    if len(maxima) < 2 or len(minima) < 2:
        return None
    # Mirror-extend end points to stabilise spline boundaries
    tmax = np.concatenate(([t[0]], t[maxima], [t[-1]]))
    xmax = np.concatenate(([x[maxima[0]]], x[maxima], [x[maxima[-1]]]))
    tmin = np.concatenate(([t[0]], t[minima], [t[-1]]))
    xmin = np.concatenate(([x[minima[0]]], x[minima], [x[minima[-1]]]))
    up = CubicSpline(tmax, xmax, bc_type="natural")(t)
    lo = CubicSpline(tmin, xmin, bc_type="natural")(t)
    m = 0.5 * (up + lo)
    return x - m


def emd(x, n_imfs=6, tol=1e-3, max_sift=15):
    t = np.arange(len(x), dtype=float)
    imfs = []
    r = x.copy()
    for _ in range(n_imfs):
        h = r.copy()
        for _ in range(max_sift):
            h_new = _sift_once(t, h)
            if h_new is None: break
            if np.linalg.norm(h_new - h) / (np.linalg.norm(h) + 1e-12) < tol:
                h = h_new; break
            h = h_new
        imfs.append(h)
        r = r - h
        if _extrema(r)[0].size < 2 or _extrema(r)[1].size < 2:
            break
    return imfs, r


if __name__ == "__main__":
    print("=== Empirical Mode Decomposition (Huang et al 1998) ===\n")
    rng = np.random.default_rng(0)
    N = 800
    t = np.linspace(0, 10, N)
    # Three-component non-stationary signal
    f1 = np.sin(2 * np.pi * 1.0 * t)                            # 1 Hz sine
    f2 = 0.5 * np.sin(2 * np.pi * 3.0 * t) * (1 + 0.5 * t / 10)  # 3 Hz amp-mod
    trend = 0.3 * t
    x = f1 + f2 + trend + 0.05 * rng.normal(size=N)

    imfs, r = emd(x, n_imfs=5)
    print(f"  Signal length {N}; extracted {len(imfs)} IMFs + residue.")
    for i, imf in enumerate(imfs):
        # Estimate dominant freq via zero crossings
        n_zc = int(((imf[:-1] * imf[1:]) < 0).sum())
        est_freq = 0.5 * n_zc / (t[-1] - t[0])
        print(f"    IMF {i+1}: mean = {imf.mean():+.4f}   dominant freq ~ {est_freq:.2f} Hz   "
              f"amplitude = {imf.std():.3f}")
    print(f"    Residue: mean = {r.mean():+.4f}   std = {r.std():.3f}   "
          f"(should track the slow trend)")

    print("\n--- library cross-check (Rlibeemd R; PyEMD / EMD-signal Python) ---")
