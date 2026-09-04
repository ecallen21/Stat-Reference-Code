"""Multivariate control charts — Hotelling T^2 (Reference Sec 37.4).

Hotelling (1947). For p-dim observations x_i, monitor

  T_i^2 = (x_i - mu_0)' Sigma_0^-1 (x_i - mu_0)   ~   F-distributed under H_0.

For individual observations (subgroup n=1), the exact limit is derived
from the beta distribution (Tracy-Young-Mason 1992). For subgroups,
use the F distribution:

  UCL = ((n-1)(m-1)) / (mn - m - n + 1) * p * F_{alpha, p, mn - m - n + 1}.

Simplification: report the phase-II sample T^2 chart with limit
p (n - 1) / (n - p) F_{alpha, p, n - p}.

Multivariate SPC catches out-of-control conditions that univariate
charts on individual variables miss.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

from scipy.stats import f as _f, chi2 as _chi2


def hotelling_t2(X, mu0=None, Sigma0=None):
    """X: (n, p). If mu0 / Sigma0 not given, use sample estimates."""
    n, p = X.shape
    if mu0 is None: mu0 = X.mean(axis=0)
    if Sigma0 is None: Sigma0 = np.cov(X, rowvar=False, ddof=1)
    S_inv = np.linalg.inv(Sigma0)
    diffs = X - mu0
    T2 = np.sum(diffs @ S_inv * diffs, axis=1)
    return T2


def t2_ucl(n, p, alpha=0.005):
    """Phase-II UCL for individual observations."""
    fq = _f.ppf(1 - alpha, p, n - p)
    return p * (n + 1) * (n - 1) / (n * (n - p)) * fq


if __name__ == "__main__":
    print("=== Hotelling T-squared multivariate control chart ===\n")
    rng = np.random.default_rng(0)
    n_baseline = 100
    p = 3
    mu_true = np.zeros(p)
    Sigma_true = np.array([[1.0, 0.6, 0.3],
                            [0.6, 1.0, 0.4],
                            [0.3, 0.4, 1.0]])
    L = np.linalg.cholesky(Sigma_true)
    baseline = rng.normal(0, 1, (n_baseline, p)) @ L.T + mu_true

    # Compute Phase I center + covariance
    mu_hat = baseline.mean(axis=0); Sigma_hat = np.cov(baseline, rowvar=False, ddof=1)

    # Phase II: injectd out-of-control observations that shift x_2 only.
    n_test = 20
    normal_obs = rng.normal(0, 1, (n_test, p)) @ L.T + mu_true
    ooc_obs = rng.normal(0, 1, (10, p)) @ L.T + np.array([0.0, 2.0, 0.0])
    X_test = np.vstack([normal_obs, ooc_obs])

    T2 = hotelling_t2(X_test, mu_hat, Sigma_hat)
    ucl = t2_ucl(n_baseline, p, alpha=0.005)
    print(f"  Phase-I baseline n = {n_baseline}, p = {p}")
    print(f"  Hotelling T^2 UCL (alpha = 0.005) = {ucl:.3f}\n")
    for i, val in enumerate(T2):
        flag = "***" if val > ucl else "   "
        stage = "in-control" if i < n_test else "SHIFTED  "
        print(f"    obs {i:>2} ({stage}) T^2 = {val:>7.3f}  {flag}")

    # Compare to univariate: individual-variable range checks
    ranges_ooc = ooc_obs.max(axis=0) - ooc_obs.min(axis=0)
    print(f"\n  Univariate ranges of shifted obs (vs baseline ranges):")
    for j in range(p):
        base_r = baseline[:, j].max() - baseline[:, j].min()
        print(f"    x_{j}   baseline range = {base_r:.2f}   ooc range = {ranges_ooc[j]:.2f}")

    print("\n--- library cross-check (R qcc::mqcc; MSQC; Python multivariate-quality-control) ---")
