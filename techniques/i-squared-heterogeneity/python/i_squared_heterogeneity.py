"""I^2 Heterogeneity Statistic (Ref Sec 47.272).

Higgins & Thompson 2002 Stat Med; Higgins et al 2003 BMJ. The
percentage of total variability in meta-analysis attributable
to BETWEEN-STUDY heterogeneity (rather than sampling error):

    I^2 = max(0, (Q - df) / Q) * 100%
    tau^2 = between-study variance
    H^2 = Q / df = 1 + tau^2 / typical_v_k

Interpretation (Cochrane Handbook): I^2 = 25% low, 50% moderate,
75% high heterogeneity. Complementary to tau^2 (absolute) and
the Q test (existence).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def i_squared(y, v):
    """Return dict of Q, df, p-value, I^2, H^2, tau^2 (DL)."""
    from scipy.stats import chi2
    y = np.asarray(y); v = np.asarray(v)
    w = 1.0 / v; K = len(y)
    ybar = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - ybar) ** 2))
    df = K - 1
    p = float(1 - chi2.cdf(Q, df))
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    H2 = Q / df if df > 0 else np.nan
    tau2 = max(0.0, (Q - df) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    return dict(Q=Q, df=df, p=p, I2=I2, H2=H2, tau2=tau2)


def prediction_interval(y, v, alpha=0.05):
    """95% prediction interval for TRUE effect in a NEW study."""
    from scipy.stats import t as tdist
    res = i_squared(y, v)
    w_re = 1.0 / (v + res["tau2"])
    ybar_re = np.sum(w_re * y) / np.sum(w_re)
    se_re = 1.0 / np.sqrt(np.sum(w_re))
    K = len(y)
    t_crit = tdist.ppf(1 - alpha / 2, df=K - 2)
    half = t_crit * np.sqrt(se_re ** 2 + res["tau2"])
    return float(ybar_re - half), float(ybar_re + half)


if __name__ == "__main__":
    print("=== I^2 Heterogeneity (Higgins & Thompson 2002) ===\n")
    rng = np.random.default_rng(0)

    for label, tau_true in [("homogeneous (tau=0.0)", 0.0),
                             ("moderate (tau=0.2)",   0.2),
                             ("high (tau=0.4)",       0.4)]:
        K = 15
        theta = rng.normal(0.5, tau_true, K)
        v = 1.0 / rng.integers(50, 300, K)
        y = np.array([rng.normal(t, np.sqrt(vv)) for t, vv in zip(theta, v)])
        r = i_squared(y, v); pi = prediction_interval(y, v)
        print(f"  {label}")
        print(f"    Q = {r['Q']:.2f}   df = {r['df']}   Q-test p = {r['p']:.4f}")
        print(f"    I^2 = {r['I2']*100:.1f}%   H^2 = {r['H2']:.2f}   tau^2 = {r['tau2']:.4f}")
        print(f"    95% prediction interval for true effect in new study: ({pi[0]:.3f}, {pi[1]:.3f})")
        print()

    print("  Cochrane thresholds: I^2 < 25% low, 25-50% moderate, > 50% substantial, > 75% considerable.")
    print("  Prediction interval is often MUCH wider than the CI of the mean - reflects")
    print("  where a new study's true effect is likely to fall, not just the pooled average.")

    print("\n--- library cross-check (metafor::rma R; meta::metagen R; PythonMeta) ---")
