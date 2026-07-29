"""Structural breaks + interrupted time series (Reference §13.7, §13.10).

CHOW TEST (§13.10) at a KNOWN break time t*:
    Split the series at t*. Fit a linear regression on each half separately
    (SSR_1 + SSR_2) and on the whole series (SSR_pooled).
    F = ((SSR_pooled - (SSR_1 + SSR_2)) / k) / ((SSR_1 + SSR_2) / (n - 2k))
    where k is the number of regression parameters.
    Small p => structural break at t*.

BAI-PERRON single-change-point scan (§13.10):
    Try every candidate break, run Chow test at each, pick the maximum-F break.
    Requires simulation-based critical values for the "sup F" test; here we
    just report the max-F point and its (approximate) F p-value.

INTERRUPTED TIME SERIES (ITS, §13.7):
    Regression with a KNOWN intervention time t*, allowing both a LEVEL shift
    and a SLOPE change:
        y_t  =  b0 + b1 t + b2 D_t + b3 (t - t*) D_t + eps
        D_t = 1 if t >= t*, else 0
    b2 = immediate level change; b3 = post-intervention slope change.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions


def chow_test(y, X, break_index: int) -> dict:
    """Chow test for a structural break at ``break_index``.

    Parameters
    ----------
    y : outcome (n).
    X : design (n x k) including intercept.
    break_index : first index of the SECOND regime.
    """
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float)
    n, k = X.shape
    if not (k < break_index < n - k):
        raise ValueError("break_index must leave at least k obs on each side")
    # Pooled
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    SSR_p = float(((y - X @ beta) ** 2).sum())
    # First half
    b1, *_ = np.linalg.lstsq(X[:break_index], y[:break_index], rcond=None)
    SSR_1 = float(((y[:break_index] - X[:break_index] @ b1) ** 2).sum())
    # Second half
    b2, *_ = np.linalg.lstsq(X[break_index:], y[break_index:], rcond=None)
    SSR_2 = float(((y[break_index:] - X[break_index:] @ b2) ** 2).sum())
    df_num = k
    df_den = n - 2 * k
    F = ((SSR_p - SSR_1 - SSR_2) / df_num) / ((SSR_1 + SSR_2) / df_den)
    return {"break_index": break_index,
            "SSR_pooled": SSR_p, "SSR_1": SSR_1, "SSR_2": SSR_2,
            "F_statistic": float(F), "df_num": df_num, "df_den": df_den,
            "p_value": float(stats.f.sf(F, df_num, df_den)),
            "method": "Chow test for structural break at known point"}


def bai_perron_max_F(y, X, trim: float = 0.15) -> dict:
    """Scan for the single break with maximum F. Trim the ends to avoid
    unreliable estimates near the boundaries."""
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float)
    n, k = X.shape
    lo = max(k, int(trim * n)); hi = min(n - k, int((1 - trim) * n))
    best = None
    for b in range(lo, hi):
        try:
            r = chow_test(y, X, b)
            if best is None or r["F_statistic"] > best["F_statistic"]:
                best = r
        except Exception:
            continue
    return {"best_break_index": best["break_index"] if best else None,
            "max_F": best["F_statistic"] if best else float("nan"),
            "sup_F_note": ("critical values for sup-F need simulation "
                            "(Andrews 1993; Bai-Perron); the F p-value here is a "
                            "lower bound"),
            "F_p_value_at_best": best["p_value"] if best else float("nan"),
            "method": "Bai-Perron single-break scan (max-F)"}


def interrupted_time_series(y, break_time: int) -> dict:
    """Interrupted time series regression with level + slope change at break_time.

    y_t = b0 + b1 t + b2 D_t + b3 (t - break_time) D_t + eps
    """
    y = np.asarray(y, dtype=float); n = len(y)
    t = np.arange(n, dtype=float)
    D = (t >= break_time).astype(float)
    X = np.column_stack([np.ones(n), t, D, (t - break_time) * D])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float((resid ** 2).sum() / (n - 4))
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t_stat = beta / np.where(se > 0, se, 1e-12)
    p_val = 2 * stats.t.sf(np.abs(t_stat), n - 4)
    labels = ["intercept b0", "pre-slope b1", "level change b2", "slope change b3"]
    return {"coefficients": dict(zip(labels, beta.tolist())),
            "SE": dict(zip(labels, se.tolist())),
            "p_values": dict(zip(labels, p_val.tolist())),
            "break_time": break_time,
            "method": "interrupted time series regression (level + slope shift)"}


if __name__ == "__main__":
    rng = np.random.default_rng(43)
    n = 100; break_t = 50
    t = np.arange(n)
    # Simulate: pre-break slope 0.2, post-break level jump of 5, post-break slope 0
    y = 10 + 0.2 * t + rng.normal(0, 1, n)
    y[break_t:] += 5.0
    y[break_t:] += -0.2 * (t[break_t:] - break_t)              # slope flattens

    X_lin = np.column_stack([np.ones(n), t])
    print(f"=== Chow test at known break t = {break_t} ===")
    r = chow_test(y, X_lin, break_t)
    print(f"  F = {r['F_statistic']:.4f}, p = {r['p_value']:.4g}")

    print("\n=== Bai-Perron single-break scan ===")
    bp = bai_perron_max_F(y, X_lin, trim=0.1)
    print(f"  best break index: {bp['best_break_index']}   (true = {break_t})")
    print(f"  max F = {bp['max_F']:.4f}   (approx p = {bp['F_p_value_at_best']:.4g})")

    print("\n=== Interrupted time series ===")
    its = interrupted_time_series(y, break_time=break_t)
    for k, v in its["coefficients"].items():
        p = its["p_values"][k]
        print(f"  {k:20s}: {v:+.4f}   p = {p:.4g}")
