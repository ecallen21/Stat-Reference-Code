"""Weak-instrument robust inference -- Anderson-Rubin (Reference Sec 15.40).

Anderson & Rubin 1949; Staiger & Stock 1997; Moreira 2003. Classical
2SLS is biased and its Wald CIs undercover when the instrument is
'weak' (first-stage F < 10). Two remedies:

  1. WEAK-IV DIAGNOSTICS
       * Cragg-Donald first-stage F statistic; Stock-Yogo critical
         values determine whether IV is 'strong enough'.

  2. WEAK-IV ROBUST TESTS OF H0: beta = beta_0
       * Anderson-Rubin (1949): regress Y - X*beta_0 on Z + controls;
         F-test that Z has no explanatory power.
       * Confidence set = { beta_0 : AR-test fails to reject }.
       * Valid REGARDLESS of instrument strength.

Setup: model Y = X * beta + u,  X endogenous, Z instrument, W controls.
The AR statistic is:

    AR(beta_0) = (Y - X * beta_0)' * P_Z_perp * (Y - X * beta_0) / (Y - X * beta_0)' * M_ZW * (Y - X * beta_0) * (n - k) / L

with k = number of controls + instruments, L = number of instruments.
Under H0 and homoskedasticity, AR ~ F(L, n - k) with reasonably-
uncorrelated residuals.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def cragg_donald_f(X, Z, W=None):
    """First-stage F for a single endogenous X with instruments Z (+ controls W)."""
    n = len(X)
    if W is None:
        W = np.ones((n, 1))
    else:
        W = np.c_[np.ones(n), W]
    #  Residualise X on W, Z on W
    def resid(A, B):
        b, *_ = np.linalg.lstsq(B, A, rcond=None)
        return A - B @ b
    X_r = resid(X, W)
    Z_r = np.column_stack([resid(Z[:, k], W) for k in range(Z.shape[1])])
    #  Regression X_r on Z_r
    pi, *_ = np.linalg.lstsq(Z_r, X_r, rcond=None)
    x_hat = Z_r @ pi
    ss_expl = np.sum(x_hat ** 2)
    ss_resid = np.sum((X_r - x_hat) ** 2)
    L = Z.shape[1]
    k_w = W.shape[1]
    df_resid = n - k_w - L
    F = (ss_expl / L) / (ss_resid / df_resid)
    return F


def anderson_rubin_ci(Y, X, Z, W=None, alpha=0.05, grid=None):
    """Anderson-Rubin confidence set for scalar beta on X."""
    n = len(Y)
    if W is None:
        W = np.ones((n, 1))
    else:
        W = np.c_[np.ones(n), W]
    L = Z.shape[1]
    k_w = W.shape[1]
    df_resid = n - k_w - L
    F_crit = stats.f.ppf(1 - alpha, L, df_resid)

    def AR(b0):
        u = Y - X.ravel() * b0
        #  Regress u on Z + W, F-test on Z coefficients
        M = np.c_[W, Z]
        b_full, *_ = np.linalg.lstsq(M, u, rcond=None)
        resid_full = u - M @ b_full
        rss_full = np.sum(resid_full ** 2)
        b_r, *_ = np.linalg.lstsq(W, u, rcond=None)
        resid_r = u - W @ b_r
        rss_r = np.sum(resid_r ** 2)
        F = (rss_r - rss_full) / L / (rss_full / df_resid)
        return F

    if grid is None:
        grid = np.linspace(-2.0, 4.0, 601)
    F_vals = np.array([AR(b) for b in grid])
    accepted = grid[F_vals <= F_crit]
    ci = (accepted.min(), accepted.max()) if len(accepted) > 0 else (np.nan, np.nan)
    return {"CI": ci, "n_accepted": len(accepted), "F_crit": F_crit,
            "grid": grid, "F_vals": F_vals}


if __name__ == "__main__":
    print("=== Weak-IV robust inference: Cragg-Donald F + Anderson-Rubin CI ===\n")
    rng = np.random.default_rng(0)
    n = 2000

    #  Strong instrument scenario
    Z_strong = rng.normal(size=(n, 1))
    U = rng.normal(size=n)
    X_strong = 1.5 * Z_strong.ravel() + 0.6 * U + rng.normal(size=n)
    Y_strong = 0.8 * X_strong + U + rng.normal(size=n)
    F_s = cragg_donald_f(X_strong, Z_strong)
    ci_s = anderson_rubin_ci(Y_strong, X_strong.reshape(-1, 1), Z_strong)
    print(f"  Strong IV:  Cragg-Donald F = {F_s:.1f}   (Stock-Yogo 5% weak-IV rule: F > 16.4)")
    print(f"              AR 95% CI      = [{ci_s['CI'][0]:+.3f}, {ci_s['CI'][1]:+.3f}]   truth = +0.80")

    #  Weak instrument scenario
    Z_weak = rng.normal(size=(n, 1))
    U = rng.normal(size=n)
    X_weak = 0.05 * Z_weak.ravel() + 1.0 * U + rng.normal(size=n)
    Y_weak = 0.8 * X_weak + U + rng.normal(size=n)
    F_w = cragg_donald_f(X_weak, Z_weak)
    ci_w = anderson_rubin_ci(Y_weak, X_weak.reshape(-1, 1), Z_weak, grid=np.linspace(-5, 5, 2001))
    print(f"\n  Weak IV:    Cragg-Donald F = {F_w:.1f}   (< 10 -> weak)")
    print(f"              AR 95% CI      = [{ci_w['CI'][0]:+.3f}, {ci_w['CI'][1]:+.3f}]   truth = +0.80")
    print(f"              (AR CI honestly wide; 2SLS Wald CI would spuriously be narrow.)")

    print("\n--- library cross-check (ivreg / ivmodel R; linearmodels.IVGMM Python) ---")
