"""A/B test fundamentals (Reference Sec 44.1).

Two-sample tests comparing a control vs treatment variant on the
canonical online metrics:

  * BINARY OUTCOME (conversion): two-proportion z-test / prop.test
    yields lift + CI.
  * CONTINUOUS OUTCOME (revenue/time): Welch two-sample t-test
    with a heteroscedasticity-safe SE.

Report:
  lift        = mean_T - mean_C  (absolute) or (mean_T - mean_C)/mean_C (relative)
  95% CI      = lift +/- 1.96 * SE
  p-value     = two-sided
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def two_proportion_test(x_C, n_C, x_T, n_T):
    """Two-proportion z-test + Wald CI on the absolute lift."""
    p_C = x_C / n_C; p_T = x_T / n_T
    diff = p_T - p_C
    se = np.sqrt(p_C * (1 - p_C) / n_C + p_T * (1 - p_T) / n_T)
    z = diff / se if se > 0 else 0.0
    p_val = 2 * stats.norm.sf(abs(z))
    return {"p_C": float(p_C), "p_T": float(p_T),
            "abs_lift": float(diff), "rel_lift": float(diff / max(p_C, 1e-12)),
            "SE": float(se),
            "CI95": (float(diff - 1.96 * se), float(diff + 1.96 * se)),
            "z": float(z), "p_value": float(p_val)}


def welch_ttest(y_C, y_T):
    m_C = np.mean(y_C); m_T = np.mean(y_T)
    v_C = np.var(y_C, ddof=1); v_T = np.var(y_T, ddof=1)
    n_C = len(y_C); n_T = len(y_T)
    diff = m_T - m_C
    se = np.sqrt(v_C / n_C + v_T / n_T)
    t_stat = diff / se
    # Welch-Satterthwaite df
    df = (v_C / n_C + v_T / n_T) ** 2 / (
        (v_C / n_C) ** 2 / (n_C - 1) + (v_T / n_T) ** 2 / (n_T - 1))
    p_val = 2 * stats.t.sf(abs(t_stat), df=df)
    return {"mean_C": float(m_C), "mean_T": float(m_T),
            "abs_lift": float(diff), "rel_lift": float(diff / max(m_C, 1e-12)),
            "SE": float(se), "df": float(df),
            "CI95": (float(diff - 1.96 * se), float(diff + 1.96 * se)),
            "t": float(t_stat), "p_value": float(p_val)}


if __name__ == "__main__":
    print("=== A/B test fundamentals: binary + continuous outcomes ===\n")
    # Binary: conversion rate 5.0% -> 5.5%
    r = two_proportion_test(500, 10000, 550, 10000)
    print(f"  Conversion A/B  p_C={r['p_C']:.4f}   p_T={r['p_T']:.4f}"
          f"   abs lift={r['abs_lift']:+.4f}   rel lift={r['rel_lift']:+.2%}")
    print(f"    SE={r['SE']:.4f}   95%CI=({r['CI95'][0]:+.4f}, {r['CI95'][1]:+.4f})"
          f"   z={r['z']:.2f}   p={r['p_value']:.3e}\n")

    # Continuous: revenue per user (heavy tail)
    rng = np.random.default_rng(0)
    y_C = rng.gamma(shape=2, scale=5, size=5000)
    y_T = rng.gamma(shape=2, scale=5.4, size=5000)   # +8% mean
    r = welch_ttest(y_C, y_T)
    print(f"  Revenue A/B     mean_C={r['mean_C']:.3f}   mean_T={r['mean_T']:.3f}"
          f"   abs lift={r['abs_lift']:+.3f}   rel lift={r['rel_lift']:+.2%}")
    print(f"    SE={r['SE']:.3f}   95%CI=({r['CI95'][0]:+.3f}, {r['CI95'][1]:+.3f})"
          f"   t={r['t']:.2f}   df={r['df']:.0f}   p={r['p_value']:.3e}\n")

    print("--- library cross-check (R stats::prop.test/t.test; Python scipy.stats.ttest_ind/chi2_contingency) ---")
