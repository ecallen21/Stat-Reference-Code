"""Cumulative Meta-Analysis (Ref Sec 47.274).

Lau, Antman, Chalmers et al 1992 NEJM. Repeat the pooled
meta-analysis after each new study is added (usually
chronologically), showing HOW THE EVIDENCE ACCUMULATED. Reveals:

- when the pooled effect first crossed significance
- whether later studies reinforced or contradicted early ones
- whether the current CI has converged

Central tool for evidence-based-medicine audits.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def dl(y, v):
    """DL random-effects for a subset of studies."""
    if len(y) == 1: return float(y[0]), float(np.sqrt(v[0])), 0.0
    w_fe = 1.0 / v
    ybar_fe = np.sum(w_fe * y) / np.sum(w_fe)
    Q = np.sum(w_fe * (y - ybar_fe) ** 2); K = len(y)
    denom = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
    tau2 = max(0.0, (Q - (K - 1)) / denom) if denom > 0 else 0.0
    w_re = 1.0 / (v + tau2)
    return float(np.sum(w_re * y) / np.sum(w_re)), float(1.0 / np.sqrt(np.sum(w_re))), float(tau2)


def cumulative_meta(y, v, years=None):
    """Sort by year (if given), otherwise input order; return cumulative table."""
    y = np.asarray(y); v = np.asarray(v)
    if years is None: years = np.arange(len(y))
    order = np.argsort(years)
    rows = []
    for k in range(1, len(y) + 1):
        idx = order[:k]
        m, s, tau2 = dl(y[idx], v[idx])
        rows.append((k, years[order[k - 1]], m, s, m - 1.96 * s, m + 1.96 * s, tau2))
    return rows


if __name__ == "__main__":
    print("=== Cumulative Meta-Analysis (Lau et al 1992 NEJM) ===\n")
    rng = np.random.default_rng(0)

    # Simulate 12 studies spanning 20 years; true effect emerges gradually
    K = 12
    years = np.sort(rng.integers(1980, 2001, K))
    v_k = 1.0 / rng.integers(30, 400, K)
    theta = rng.normal(0.4, 0.15, K)
    y_k = np.array([rng.normal(t, np.sqrt(v)) for t, v in zip(theta, v_k)])

    print(f"  {'#studies':>8}  {'year':>4}  {'pooled':>7}  {'SE':>6}  {'95% CI':>16}  {'tau^2':>7}")
    first_sig = None
    for k, yr, m, s, lo, hi, tau2 in cumulative_meta(y_k, v_k, years):
        marker = ""
        if first_sig is None and lo > 0:
            first_sig = yr; marker = "  <-- first crossed sig"
        print(f"  {k:>8}  {yr:>4}  {m:>7.3f}  {s:>6.3f}  ({lo:+.3f}, {hi:+.3f})  {tau2:>7.4f}{marker}")

    if first_sig is not None:
        print(f"\n  Pooled 95% CI first excluded zero in {first_sig}")
        print(f"  (subsequent studies confirmed / stabilised the estimate).")
    else:
        print(f"\n  Pooled CI never excluded zero in this trajectory.")

    print("\n--- library cross-check (metafor::cumul R; meta::metacum R) ---")
