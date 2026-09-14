"""Hartung-Knapp-Sidik-Jonkman Adjustment (Ref Sec 47.270).

Hartung & Knapp 2001 Stat Med; Sidik & Jonkman 2002 Stat Med.
Improved inference for random-effects meta-analysis: instead
of z-quantiles, use t-quantiles with (K-1) df AND a variance
estimator that adapts to observed heterogeneity:

    V_HK = 1/(K-1) sum w_k^RE * (y_k - y_bar^RE)^2 / sum w_k^RE
    CI  = y_bar^RE +/- t_{K-1, 1-alpha/2} * sqrt(V_HK)

Recommended default for small K (< 20) and/or moderate
heterogeneity by Cochrane Handbook v6+.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def dl_estimate(y, v):
    """DerSimonian-Laird tau^2 + RE pooled estimate."""
    w_fe = 1.0 / v; ybar_fe = np.sum(w_fe * y) / np.sum(w_fe)
    Q = np.sum(w_fe * (y - ybar_fe) ** 2); K = len(y)
    tau2 = max(0.0, (Q - (K - 1)) / (np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)))
    w_re = 1.0 / (v + tau2); ybar_re = np.sum(w_re * y) / np.sum(w_re)
    return ybar_re, w_re, tau2, Q


def hksj_ci(y, v, alpha=0.05):
    """HKSJ CI: adaptive variance + t distribution."""
    from scipy.stats import t as tdist
    K = len(y)
    ybar_re, w_re, _, _ = dl_estimate(y, v)
    V_hk = np.sum(w_re * (y - ybar_re) ** 2) / (K - 1) / np.sum(w_re)
    t_crit = tdist.ppf(1 - alpha / 2, df=K - 1)
    se = np.sqrt(V_hk)
    return ybar_re, se, ybar_re - t_crit * se, ybar_re + t_crit * se


def dl_z_ci(y, v, alpha=0.05):
    """Classical DL CI with z-quantile."""
    from scipy.stats import norm
    ybar_re, w_re, _, _ = dl_estimate(y, v)
    se = 1.0 / np.sqrt(np.sum(w_re))
    z = norm.ppf(1 - alpha / 2)
    return ybar_re, se, ybar_re - z * se, ybar_re + z * se


if __name__ == "__main__":
    print("=== Hartung-Knapp-Sidik-Jonkman (HK 2001; SJ 2002) ===\n")
    rng = np.random.default_rng(0)

    # Small-K meta-analysis with moderate heterogeneity
    for K in [5, 10, 30]:
        theta_k = rng.normal(0.3, 0.25, K)
        v_k = 1.0 / rng.integers(30, 200, K)
        y_k = np.array([rng.normal(t, np.sqrt(v)) for t, v in zip(theta_k, v_k)])
        m_dl, se_dl, lo_dl, hi_dl = dl_z_ci(y_k, v_k)
        m_hk, se_hk, lo_hk, hi_hk = hksj_ci(y_k, v_k)
        print(f"  K = {K:>2}")
        print(f"    DL   pooled:  {m_dl:.3f}   SE {se_dl:.3f}   95% CI ({lo_dl:.3f}, {hi_dl:.3f})   width {hi_dl-lo_dl:.3f}")
        print(f"    HKSJ pooled:  {m_hk:.3f}   SE {se_hk:.3f}   95% CI ({lo_hk:.3f}, {hi_hk:.3f})   width {hi_hk-lo_hk:.3f}")
        print()

    # Type-I preservation: simulate under H0 (true effect = 0)
    print("  Type-I under H0 (5000 sims of K=8 meta-analyses; true effect = 0):")
    n_sim = 5000; rej_dl = 0; rej_hk = 0
    for _ in range(n_sim):
        K = 8
        theta_k = rng.normal(0, 0.25, K)                          # heterogeneous but zero mean
        v_k = 1.0 / rng.integers(30, 200, K)
        y_k = np.array([rng.normal(t, np.sqrt(v)) for t, v in zip(theta_k, v_k)])
        _, _, lo, hi = dl_z_ci(y_k, v_k);   rej_dl += int(lo > 0 or hi < 0)
        _, _, lo, hi = hksj_ci(y_k, v_k);   rej_hk += int(lo > 0 or hi < 0)
    print(f"    DL Type-I   = {rej_dl/n_sim:.4f}   (nominal 0.05; inflated under heterogeneity)")
    print(f"    HKSJ Type-I = {rej_hk/n_sim:.4f}   (closer to nominal)")

    print("\n--- library cross-check (metafor::rma(method='DL', test='knha') R; PythonMeta) ---")
