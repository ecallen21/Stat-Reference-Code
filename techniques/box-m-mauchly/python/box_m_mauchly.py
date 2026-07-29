"""Box's M and Mauchly's sphericity tests (Reference §9.3, §12.2).

Both diagnose covariance-structure assumptions for downstream analyses.

Box's M
    Null: covariance matrices S_1, ..., S_K in K groups are equal.
    Used to check the MANOVA / LDA equality-of-covariance assumption.

    M = (n - K) log|S_pooled| - sum_g (n_g - 1) log|S_g|

    Bartlett-style correction c gives approximately chi-square with
    p(p+1)(K-1)/2 df.  Box's M is highly sensitive to normality;
    with large n a "significant" M often signals departure so mild it
    does not harm MANOVA.  Report M with caution.

Mauchly's sphericity
    Null: the covariance of within-subject repeated measures has
    "spherical" form -- equal variances of pairwise DIFFERENCES.  This
    is the assumption behind the univariate repeated-measures ANOVA
    F-test.  If rejected, use Greenhouse-Geisser or Huynh-Feldt
    corrected df, or switch to a multivariate / mixed-model approach.

    Compute on the (p-1) x (p-1) contrast covariance C = M S M^T
    where M is any orthonormal contrast matrix; W = |C| / (tr(C)/(p-1))^(p-1).
    Chi-square approximation with p(p-1)/2 - 1 df.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def box_m_test(groups: list) -> dict:
    """Box's M test for equality of covariance matrices.

    groups : list of (n_g x p) arrays, one per group.
    """
    Xs = [np.asarray(g, dtype=float) for g in groups]
    K = len(Xs); p = Xs[0].shape[1]
    ns = np.array([len(x) for x in Xs])
    n = int(ns.sum())
    S_pooled = np.zeros((p, p))
    log_det_S = np.zeros(K)
    for i, X in enumerate(Xs):
        Si = np.cov(X, rowvar=False, ddof=1)
        sign, ld = np.linalg.slogdet(Si)
        log_det_S[i] = ld
        S_pooled += (len(X) - 1) * Si
    S_pooled /= (n - K)
    _, log_det_pooled = np.linalg.slogdet(S_pooled)
    M = (n - K) * log_det_pooled - np.sum((ns - 1) * log_det_S)

    # Box's chi-square correction c1
    inv_sum = np.sum(1.0 / (ns - 1))
    c1 = ((2 * p * p + 3 * p - 1) / (6 * (p + 1) * (K - 1))) * (inv_sum - 1.0 / (n - K))
    chi2_stat = (1 - c1) * M
    df = p * (p + 1) * (K - 1) // 2
    p_chi2 = float(stats.chi2.sf(chi2_stat, df))
    return {"M": float(M), "chi2_approx": float(chi2_stat), "df": int(df),
            "p_value_chi2": p_chi2,
            "n_groups": int(K), "p_dim": int(p), "n_total": int(n),
            "method": "Box's M test (equality of covariance matrices)"}


def mauchly_sphericity(X) -> dict:
    """Mauchly's sphericity test on within-subject repeated measures.

    X : n_subjects x p matrix (p repeated measures per subject).
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if p < 3: raise ValueError("need at least 3 repeated measures")
    S = np.cov(X, rowvar=False, ddof=1)
    # Helmert-style orthonormal contrast matrix M (p-1 x p)
    Mc = np.zeros((p - 1, p))
    for k in range(p - 1):
        Mc[k, :k + 1] = 1.0 / (k + 1)
        Mc[k, k + 1] = -1.0
        Mc[k] /= np.sqrt((Mc[k] ** 2).sum())
    C = Mc @ S @ Mc.T
    q = p - 1
    trC = np.trace(C)
    detC = np.linalg.det(C)
    W = detC / (trC / q) ** q
    df = q * (q + 1) // 2 - 1
    # Bartlett-corrected chi-square
    d = 1 - (2 * q * q + q + 2) / (6 * q * (n - 1))
    chi2_stat = -(n - 1) * d * math.log(max(W, 1e-300))
    p_val = float(stats.chi2.sf(chi2_stat, df))
    # Greenhouse-Geisser epsilon
    eigs = np.linalg.eigvalsh(C)
    eigs = eigs[eigs > 1e-12]
    gg_eps = float((eigs.sum() ** 2) / (q * (eigs ** 2).sum()))
    hf_eps = min(1.0, (n * q * gg_eps - 2) / (q * (n - 1 - q * gg_eps)))
    return {"W": float(W), "chi2_approx": float(chi2_stat), "df": int(df),
            "p_value_chi2": p_val,
            "greenhouse_geisser_epsilon": gg_eps,
            "huynh_feldt_epsilon": float(hf_eps),
            "n": int(n), "p_reps": int(p),
            "method": "Mauchly's sphericity test"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Box's M test (K=3, equal covariance) ===")
    Xs = [rng.multivariate_normal([0]*3, np.eye(3), 40) for _ in range(3)]
    r = box_m_test(Xs)
    print(f"  M = {r['M']:.4f}, chi2 = {r['chi2_approx']:.4f}, df = {r['df']}, p = {r['p_value_chi2']:.4f}")

    print("\n=== Box's M test (K=3, unequal covariance) ===")
    Xs = [rng.multivariate_normal([0]*3, np.eye(3), 40),
          rng.multivariate_normal([0]*3, 3 * np.eye(3), 40),
          rng.multivariate_normal([0]*3, np.diag([1, 5, 1]), 40)]
    r = box_m_test(Xs)
    print(f"  M = {r['M']:.4f}, chi2 = {r['chi2_approx']:.4f}, df = {r['df']}, p = {r['p_value_chi2']:.4f}")

    print("\n=== Mauchly's sphericity (spherical data) ===")
    X = rng.normal(0, 1, size=(30, 4))
    r = mauchly_sphericity(X)
    print(f"  W = {r['W']:.4f}, chi2 = {r['chi2_approx']:.4f}, p = {r['p_value_chi2']:.4f}")
    print(f"  GG epsilon = {r['greenhouse_geisser_epsilon']:.3f}, HF = {r['huynh_feldt_epsilon']:.3f}")

    print("\n=== Mauchly's sphericity (non-spherical - AR(1) within-subject) ===")
    n, p = 40, 4
    rho = 0.85
    Sigma = np.array([[rho ** abs(i - j) for j in range(p)] for i in range(p)])
    X = rng.multivariate_normal(np.zeros(p), Sigma, n)
    r = mauchly_sphericity(X)
    print(f"  W = {r['W']:.4f}, chi2 = {r['chi2_approx']:.4f}, p = {r['p_value_chi2']:.4f}")
    print(f"  GG epsilon = {r['greenhouse_geisser_epsilon']:.3f}, HF = {r['huynh_feldt_epsilon']:.3f}")
