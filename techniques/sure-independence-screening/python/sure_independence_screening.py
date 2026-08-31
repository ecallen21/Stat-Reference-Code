"""Sure Independence Screening (Reference Sec 32.7).

Fan & Lv (2008) 'Sure independence screening for ultrahigh dimensional
feature space.'

For p >> n (millions of features): plain LASSO / SCAD is intractable.
SIS pre-screens features by their MARGINAL correlation with y:

  omega_j = |corr(X_j, y)|,  j = 1..p

Keep the top d_n features (typically d_n = n / log(n) or n - 1). Under
mild conditions the true active set is retained with probability -> 1
(the SURE-SCREENING property).

Iterative SIS (ISIS) applies SIS + regularised regression + rescreening
on residuals.

Here we screen p = 5000 features down to d_n = 20 on n = 200 obs, then
compare downstream LASSO fits with and without screening.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sis(X, y, d_n):
    """Rank features by |corr(X_j, y)|; return top d_n indices."""
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    corrs = np.abs((Xc.T @ yc) / (np.sqrt((Xc ** 2).sum(axis=0) + 1e-12)
                                    * np.sqrt((yc ** 2).sum() + 1e-12)))
    return np.argsort(-corrs)[:d_n], corrs


def _soft(x, lam): return np.sign(x) * np.maximum(0, np.abs(x) - lam)


def lasso_cd(X, y, lam, max_iter=200):
    n, d = X.shape
    beta = np.zeros(d)
    XtX_diag = (X ** 2).sum(axis=0) / n
    for _ in range(max_iter):
        r = y - X @ beta
        for j in range(d):
            xj = X[:, j]
            rho = xj @ r / n + XtX_diag[j] * beta[j]
            b_new = _soft(rho, lam) / max(XtX_diag[j], 1e-12)
            r = r + xj * (beta[j] - b_new)
            beta[j] = b_new
    return beta


if __name__ == "__main__":
    print("=== Sure Independence Screening (Fan-Lv 2008) ===\n")
    rng = np.random.default_rng(0)
    n, p = 200, 5000
    X = rng.normal(0, 1, (n, p))
    true_supp = np.array([10, 100, 1000, 2500, 4700])
    beta_true = np.zeros(p)
    beta_true[true_supp] = [3.0, -2.5, 2.0, 1.8, -1.5]
    y = X @ beta_true + rng.normal(0, 0.5, n)

    d_n = int(n / np.log(n)) + 5
    idx_sis, corrs = sis(X, y, d_n)
    tp_screen = int(np.isin(true_supp, idx_sis).sum())
    print(f"  d = {p}   n = {n}   screening size d_n = {d_n}")
    print(f"  true features retained after SIS: {tp_screen}/5")

    # Compare: LASSO on all p features vs LASSO on screened subset.
    import time
    t0 = time.perf_counter()
    beta_full = lasso_cd(X, y, lam=0.10)
    t_full = time.perf_counter() - t0

    X_screen = X[:, idx_sis]
    t0 = time.perf_counter()
    beta_screen = lasso_cd(X_screen, y, lam=0.10)
    t_screen = time.perf_counter() - t0

    # Support recovery
    supp_full = set(np.where(np.abs(beta_full) > 1e-3)[0])
    supp_screen_local = set(np.where(np.abs(beta_screen) > 1e-3)[0])
    supp_screen = set(int(idx_sis[i]) for i in supp_screen_local)
    tp_full = len(supp_full & set(true_supp.tolist()))
    tp_screen_lasso = len(supp_screen & set(true_supp.tolist()))
    print(f"\n  LASSO on all features:       TP={tp_full}/5   FP={len(supp_full) - tp_full}   time {t_full:.3f}s")
    print(f"  LASSO after SIS screening:   TP={tp_screen_lasso}/5   FP={len(supp_screen) - tp_screen_lasso}   time {t_screen:.3f}s")
    print(f"  speedup: {t_full / max(t_screen, 1e-6):.1f}x\n")

    print("--- library cross-check (R SIS package; Python nihao or celer with screening) ---")
