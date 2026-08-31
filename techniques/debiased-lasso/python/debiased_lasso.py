"""Debiased LASSO (Reference Sec 32.4).

Zhang & Zhang (2014); van de Geer, Buhlmann, Ritov & Dezeure (2014):
'On asymptotically optimal confidence regions and tests for high-
dimensional models.'

LASSO produces a BIASED estimator due to shrinkage. Debiased LASSO
adds a CORRECTION so that each coefficient has an approximately
Gaussian distribution centred at the true value, enabling valid CIs
and p-values even when p >> n.

Formula:  beta_debiased = beta_lasso + (1/n) M X' (y - X beta_lasso)
where M approximates the inverse of Sigma = X' X / n via NODE-WISE
LASSO regression: for each j, regress X_j on X_{-j} with LASSO, then
form m_j from the residual scaling.

Under mild sparsity + eigenvalue conditions:
  sqrt(n) (beta_debiased_j - beta_true_j)  -> N(0, sigma_j^2)
so CIs = beta_debiased_j +/- z_{1-alpha/2} * sigma_hat_j / sqrt(n).

Here we implement a compact debiased-LASSO with node-wise LASSO for M
and check empirical CI coverage on synthetic sparse data.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _soft(x, lam): return np.sign(x) * np.maximum(0, np.abs(x) - lam)


def _lasso_cd(X, y, lam, max_iter=200, tol=1e-8):
    n, d = X.shape
    beta = np.zeros(d)
    XtX_diag = (X ** 2).sum(axis=0) / n
    for _ in range(max_iter):
        beta_old = beta.copy()
        r = y - X @ beta
        for j in range(d):
            xj = X[:, j]
            rho = xj @ r / n + XtX_diag[j] * beta[j]
            beta_j_new = _soft(rho, lam) / max(XtX_diag[j], 1e-12)
            r = r + xj * (beta[j] - beta_j_new)
            beta[j] = beta_j_new
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta


def _nodewise_lasso(X, lam_M):
    """For each j, regress X_j on X_{-j} with LASSO; return M."""
    n, d = X.shape
    M = np.zeros((d, d))
    for j in range(d):
        others = [k for k in range(d) if k != j]
        gamma = _lasso_cd(X[:, others], X[:, j], lam_M)
        # Residual for column j
        resid = X[:, j] - X[:, others] @ gamma
        tau2 = float(resid @ resid) / n + lam_M * float(np.sum(np.abs(gamma)))
        M[j, j] = 1.0 / tau2
        for k, ok in enumerate(others):
            M[j, ok] = -gamma[k] / tau2
    return M


def debiased_lasso(X, y, lam=0.10, lam_M=0.10):
    beta_lasso = _lasso_cd(X, y, lam)
    M = _nodewise_lasso(X, lam_M)
    n = X.shape[0]
    r = y - X @ beta_lasso
    beta_d = beta_lasso + (M @ X.T @ r) / n
    # Approximate variance for each debiased coef
    sigma_hat2 = float(r @ r / n)
    var_beta = sigma_hat2 * np.diag(M @ (X.T @ X / n) @ M.T) / n
    return beta_d, np.sqrt(var_beta), beta_lasso


if __name__ == "__main__":
    print("=== Debiased LASSO (Zhang-Zhang 2014, van de Geer 2014) ===\n")
    rng = np.random.default_rng(0)
    n, d = 200, 40
    n_trials = 100
    covered = np.zeros(d)
    lengths = np.zeros(d)
    beta_true = np.zeros(d)
    beta_true[[0, 5, 10]] = [1.5, -1.0, 2.0]
    for trial in range(n_trials):
        X = rng.normal(0, 1, (n, d))
        y = X @ beta_true + rng.normal(0, 0.5, n)
        beta_d, se, _ = debiased_lasso(X, y, lam=0.1, lam_M=0.15)
        lo = beta_d - 1.96 * se
        hi = beta_d + 1.96 * se
        covered += ((beta_true >= lo) & (beta_true <= hi)).astype(int)
        lengths += (hi - lo)
    coverage = covered / n_trials
    avg_length = lengths / n_trials

    print(f"  Empirical 95% coverage over {n_trials} trials:")
    print(f"    signal coord 0 (true 1.5): {coverage[0]:.3f}   avg CI width {avg_length[0]:.3f}")
    print(f"    signal coord 5 (true -1.0): {coverage[5]:.3f}   avg CI width {avg_length[5]:.3f}")
    print(f"    signal coord 10 (true 2.0): {coverage[10]:.3f}   avg CI width {avg_length[10]:.3f}")
    print(f"    mean coverage on 37 zeros: {coverage[[k for k in range(d) if k not in [0,5,10]]].mean():.3f}")
    print(f"    mean CI width on zeros:    {avg_length[[k for k in range(d) if k not in [0,5,10]]].mean():.3f}\n")

    print("--- library cross-check (R hdi::lasso.proj, R selectiveInference; Python celer/hdlasso) ---")
