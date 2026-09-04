"""Calibration plots (Reference Sec 39.19).

Plot OBSERVED event rate vs PREDICTED probability -- perfect
calibration = 45-degree line.

Two constructions:

  A. GROUPED (decile) plot: bin predictions into g groups; plot
     observed rate vs mean predicted per group.  Simple; sensitive
     to bin count.

  B. SMOOTHED (LOESS or spline) plot: nonparametric smoother of y
     on p_hat; less arbitrary than binning.  Steyerberg 2019
     recommends this.

Numerical summaries beside the plot: ICI, E-max, E-90.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def grouped_calibration(y, p, g=10):
    """Group into g deciles by predicted probability; return per-group means."""
    order = np.argsort(p)
    groups = np.array_split(order, g)
    rows = []
    for i, gi in enumerate(groups):
        if len(gi) == 0:
            continue
        rows.append({"group": i + 1, "n": int(len(gi)),
                     "pred_mean": float(p[gi].mean()),
                     "obs_rate": float(y[gi].mean())})
    return rows


def loess_calibration(y, p, span=0.75):
    """Local-linear (LOWESS-style) calibration curve at unique predicted probabilities.

    Simple tricube-weighted local regression.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(p)
    k = max(int(np.ceil(span * n)), 5)
    order = np.argsort(p)
    p_s = p[order]; y_s = y[order]
    xs = np.linspace(p_s.min(), p_s.max(), 50)
    fitted = []
    for x0 in xs:
        d = np.abs(p_s - x0)
        h = np.partition(d, k - 1)[k - 1]
        h = max(h, 1e-9)
        w = (1 - np.clip(d / h, 0, 1) ** 3) ** 3
        # Weighted linear fit around x0
        W = np.diag(w)
        X = np.column_stack([np.ones_like(p_s), p_s - x0])
        beta, *_ = np.linalg.lstsq(W @ X, W @ y_s, rcond=None)
        fitted.append(beta[0])
    return list(zip(xs.tolist(), fitted))


def calibration_indices(y, p):
    """ICI + E-max + E-90 from a LOESS calibration curve."""
    curve = loess_calibration(y, p)
    xs, fits = zip(*curve)
    xs = np.asarray(xs); fits = np.asarray(fits)
    # Interpolate at each observation's predicted probability
    interp = np.interp(p, xs, fits)
    diffs = np.abs(interp - p)
    ici = float(diffs.mean())
    e_max = float(diffs.max())
    e_90 = float(np.quantile(diffs, 0.90))
    return {"ICI": ici, "E_max": e_max, "E_90": e_90}


if __name__ == "__main__":
    print("=== Calibration plots: grouped + LOESS + ICI/E-max/E-90 ===\n")
    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)

    # Well-calibrated
    p_good = 1 / (1 + np.exp(-x))
    # Miscalibrated: over-confident (extremes exaggerated)
    p_bad = np.clip(1 / (1 + np.exp(-1.6 * x)), 0.01, 0.99)

    for name, p in [("well-calibrated", p_good), ("over-confident", p_bad)]:
        print(f"  Model: {name}")
        rows = grouped_calibration(y, p, g=10)
        print(f"    {'group':>5s} {'n':>4s} {'pred_mean':>10s} {'obs_rate':>10s}")
        for r in rows:
            print(f"    {r['group']:>5d} {r['n']:>4d} {r['pred_mean']:>10.3f} {r['obs_rate']:>10.3f}")
        idx = calibration_indices(y, p)
        print(f"    LOESS-based ICI = {idx['ICI']:.3f}   E-max = {idx['E_max']:.3f}"
              f"   E-90 = {idx['E_90']:.3f}\n")

    print("--- library cross-check (R rms::calibrate; Python sklearn.calibration.calibration_curve) ---")
