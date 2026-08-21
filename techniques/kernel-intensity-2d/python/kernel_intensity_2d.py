"""2D kernel intensity estimation for spatial point patterns (Reference §23.14).

Estimated intensity (points per unit area) at location s:

    lambda_hat(s) = (1 / edge(s)) * sum_i K_h(s - x_i)

with a Gaussian kernel and bandwidth h.  Border edge correction divides by
the integral of K_h over the study window at s (Diggle correction);
without correction, intensity drops artificially near the boundary.

Contrast with density KDE: an intensity has units of *counts per area*
(sums to n over the window), whereas a KDE density sums to 1.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math (sqrt, pi, log)

import numpy as np    # numerical arrays + linear algebra


def _gauss2(dx, dy, h):
    return np.exp(-0.5 * (dx * dx + dy * dy) / (h * h)) / (2 * math.pi * h * h)


def kernel_intensity(pts, xgrid, ygrid, bandwidth: float,
                     window=None, edge_correct: bool = True) -> dict:
    """Return lambda_hat on the mesh xgrid x ygrid."""
    pts = np.asarray(pts, dtype=float)
    xs, ys = np.meshgrid(xgrid, ygrid)
    lam = np.zeros_like(xs, dtype=float)
    for (px, py) in pts:
        lam += _gauss2(xs - px, ys - py, bandwidth)
    if edge_correct and window is not None:
        xmin, xmax, ymin, ymax = window
        # integral of the kernel over the window at each grid point:
        # for Gaussian, the integral is a product of 1D normal CDF differences
        from scipy.special import erf as _erf                 # vectorised erf
        def _phi(z):                                          # standard normal CDF
            return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
        cx = _phi((xmax - xs) / bandwidth) - _phi((xmin - xs) / bandwidth)
        cy = _phi((ymax - ys) / bandwidth) - _phi((ymin - ys) / bandwidth)
        lam = lam / (cx * cy)
    return {"xgrid": np.asarray(xgrid, dtype=float),
            "ygrid": np.asarray(ygrid, dtype=float),
            "lambda_hat": lam, "bandwidth": bandwidth,
            "method": "2D Gaussian kernel intensity"
                       + (" (Diggle edge correction)" if edge_correct else "")}


def scott_bandwidth(pts) -> float:
    """Scott's rule for 2D: h = n^(-1/6) * mean of coordinate SDs."""
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    sd = pts.std(axis=0, ddof=1).mean()
    return float(sd * n ** (-1.0 / 6.0))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    window = (0.0, 10.0, 0.0, 10.0)
    # mixture: cluster centred at (3, 7) + uniform background
    n_clust = 60; n_back = 40
    clust = np.column_stack([rng.normal(3, 0.6, n_clust), rng.normal(7, 0.6, n_clust)])
    back = np.column_stack([rng.uniform(0, 10, n_back), rng.uniform(0, 10, n_back)])
    pts = np.clip(np.vstack([clust, back]), 0.01, 9.99)
    n = len(pts)

    h = scott_bandwidth(pts)
    print(f"=== Kernel intensity (Gaussian) ===")
    print(f"  n = {n},  Scott bandwidth h = {h:.3f}")

    gx = np.linspace(0.25, 9.75, 20)
    gy = np.linspace(0.25, 9.75, 20)
    fit = kernel_intensity(pts, gx, gy, bandwidth=h, window=window, edge_correct=True)

    # integrated intensity should equal n (approx)
    dx = gx[1] - gx[0]; dy = gy[1] - gy[0]
    total = float(fit["lambda_hat"].sum() * dx * dy)
    print(f"  integrated lambda_hat over window ≈ {total:.2f}   (expected ≈ n = {n})")

    # peak location
    ij = np.unravel_index(np.argmax(fit["lambda_hat"]), fit["lambda_hat"].shape)
    peak_x = fit["xgrid"][ij[1]]; peak_y = fit["ygrid"][ij[0]]
    print(f"  peak intensity at ({peak_x:.2f}, {peak_y:.2f})   "
          f"true cluster centre = (3.00, 7.00)")
    print(f"  peak lambda = {fit['lambda_hat'][ij]:.3f} per unit area")
    print(f"  background lambda_hat at (5, 5) ≈ "
          f"{fit['lambda_hat'][np.argmin(np.abs(gy-5)), np.argmin(np.abs(gx-5))]:.3f}")

    print("\n--- library cross-check (R spatstat::density.ppp; MASS::kde2d) ---")
