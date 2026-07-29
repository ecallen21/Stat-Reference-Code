"""Multivariate outlier detection (Reference §9.6, §9.7).

Two complementary tools:

1) Mahalanobis distance
    D2_i = (x_i - mean)^T Sigma^-1 (x_i - mean)
    Under multivariate normality, D2 ~ chi-square with p df.
    Flag i as outlier if D2_i > chi2_{1-alpha, p}.

    Weakness: mean and Sigma are themselves DRAGGED by outliers
    (the "masking" problem).  A cluster of outliers inflates the
    covariance so its own distance shrinks.

2) Minimum Covariance Determinant (MCD, Rousseeuw 1985)
    Robust estimate of (mean, Sigma) that resists a fraction of
    contamination up to (n - h) / n, where h ~ 0.75 n is the
    subsample size chosen to MINIMIZE the determinant of the
    subsample covariance.  Robust Mahalanobis distances computed
    from (mu_MCD, S_MCD) reveal outliers that classical distances
    mask.

Implementation notes
    We use a simple FAST-MCD-flavored algorithm: many random subsets of
    size h, C-step to convergence, keep the one with smallest |S|.
    Not identical to Rousseeuw & Van Driessen (1999) FAST-MCD but
    delivers the same qualitative behaviour on small demos.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def mahalanobis_outliers(X, alpha: float = 0.025) -> dict:
    """Classical Mahalanobis distance and outlier flags."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    mu = X.mean(0)
    S = np.cov(X, rowvar=False, ddof=1)
    Sinv = np.linalg.pinv(S)
    d = X - mu
    D2 = np.einsum("ij,jk,ik->i", d, Sinv, d)
    cutoff = float(stats.chi2.ppf(1 - alpha, p))
    return {"D2": D2, "cutoff_chi2": cutoff,
            "outlier_idx": np.where(D2 > cutoff)[0].tolist(),
            "n": int(n), "p": int(p), "alpha": alpha,
            "method": "Classical Mahalanobis distance"}


def _c_step(X, subset):
    """One C-step: recompute mu/S on subset, pick h smallest D2."""
    n, p = X.shape; h = len(subset)
    mu = X[subset].mean(0)
    S = np.cov(X[subset], rowvar=False, ddof=1)
    Sinv = np.linalg.pinv(S)
    d = X - mu
    D2 = np.einsum("ij,jk,ik->i", d, Sinv, d)
    new_subset = np.argsort(D2)[:h]
    sign, logdet = np.linalg.slogdet(S)
    return new_subset, logdet, mu, S


def mcd_outliers(X, alpha: float = 0.025, h_frac: float = 0.75,
                 n_starts: int = 200, seed: int = 0) -> dict:
    """Fast-MCD-flavored robust Mahalanobis-based outlier detection."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    h = max(p + 1, int(np.ceil(h_frac * n)))
    rng = np.random.default_rng(seed)
    best_logdet = np.inf; best_mu = None; best_S = None
    for _ in range(n_starts):
        idx = rng.choice(n, size=p + 1, replace=False)
        mu0 = X[idx].mean(0)
        S0 = np.cov(X[idx], rowvar=False, ddof=1) + 1e-6 * np.eye(p)
        d = X - mu0
        D2 = np.einsum("ij,jk,ik->i", d, np.linalg.pinv(S0), d)
        subset = np.argsort(D2)[:h]
        logdet = np.inf
        for _ in range(30):
            new_subset, new_logdet, mu, S = _c_step(X, subset)
            if new_logdet >= logdet - 1e-10: break
            subset, logdet = new_subset, new_logdet
        if logdet < best_logdet:
            best_logdet = logdet; best_mu = mu; best_S = S
    # Consistency correction: rescale so subsample covariance is unbiased under normality
    # Rousseeuw-Van Aelst: c = median(D2_MCD) / chi2_{0.5, p}
    d = X - best_mu
    D2_raw = np.einsum("ij,jk,ik->i", d, np.linalg.pinv(best_S), d)
    med = float(np.median(D2_raw))
    c = med / stats.chi2.ppf(0.5, p) if med > 0 else 1.0
    S_rob = best_S * c
    Sinv = np.linalg.pinv(S_rob)
    d = X - best_mu
    D2 = np.einsum("ij,jk,ik->i", d, Sinv, d)
    cutoff = float(stats.chi2.ppf(1 - alpha, p))
    return {"D2_robust": D2, "cutoff_chi2": cutoff,
            "outlier_idx": np.where(D2 > cutoff)[0].tolist(),
            "mu_MCD": best_mu, "S_MCD": S_rob, "h": int(h),
            "n": int(n), "p": int(p), "alpha": alpha,
            "method": "Robust Mahalanobis via MCD"}


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n, p = 100, 2
    X_clean = rng.multivariate_normal([0, 0], np.eye(p), n - 8)
    X_out = rng.multivariate_normal([6, 6], 0.3 * np.eye(p), 8)
    X = np.vstack([X_clean, X_out])
    true_out = list(range(n - 8, n))

    print("=== Classical Mahalanobis (masked by outlier cluster) ===")
    r = mahalanobis_outliers(X, alpha=0.025)
    print(f"  flagged: {r['outlier_idx']}")
    print(f"  true out: {true_out}")

    print("\n=== Robust MCD-based Mahalanobis ===")
    r = mcd_outliers(X, alpha=0.025, n_starts=100, seed=0)
    print(f"  flagged: {sorted(r['outlier_idx'])}")
    print(f"  true out: {true_out}")

    print("\n--- library cross-check (scikit-learn MinCovDet) ---")
    try:
        from sklearn.covariance import MinCovDet
        mcd = MinCovDet(random_state=0).fit(X)
        d2 = mcd.mahalanobis(X)
        cutoff = stats.chi2.ppf(0.975, p)
        print(f"  sklearn flagged: {sorted(np.where(d2 > cutoff)[0].tolist())}")
    except Exception as ex:
        print(f"  (sklearn not available: {ex})")
