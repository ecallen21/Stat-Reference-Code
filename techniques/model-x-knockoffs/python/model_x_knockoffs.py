"""Model-X knockoffs (Reference Sec 32.5).

Candes, Fan, Janson & Lv (2018) 'Panning for gold: Model-X knockoffs
for high-dimensional controlled variable selection.'

Construct 'knockoff' copies X_tilde of the design X that MIMIC the
correlation structure of X but are conditionally independent of Y
given X. Fit a statistic Z_j (e.g. LASSO absolute-coefficient) for
each original + knockoff pair, and

  W_j = Z_j(original) - Z_j(knockoff).

Selection: pick features with W_j > threshold, choosing threshold so
that #{j : W_j <= -tau} / #{j : W_j >= tau} <= q  (FDR level q).

For Gaussian X ~ N(0, Sigma), construct knockoffs via:

  X_tilde = X * (I - Sigma^{-1} diag(s))
             + Chol(2 diag(s) - diag(s) Sigma^{-1} diag(s)) * Z_new
  where s minimises correlation between (X_j, X_tilde_j) subject to
  being feasible (equi-correlated: s_j = min(1, 2 lambda_min(Sigma))).

Here we implement equi-correlated Gaussian knockoffs + LASSO importance,
verify FDR control on synthetic data.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _lasso_coefs(X, y, lam):
    from sklearn.linear_model import Lasso
    return Lasso(alpha=lam, fit_intercept=False, max_iter=5000).fit(X, y).coef_


def equi_gaussian_knockoffs(X, seed=0):
    """Equi-correlated Gaussian knockoffs (Candes 2018)."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    Sigma = X.T @ X / n
    # Enforce positive-definiteness
    Sigma = Sigma + 1e-4 * np.eye(p)
    Sigma_inv = np.linalg.inv(Sigma)
    lam_min = float(np.linalg.eigvalsh(Sigma).min())
    s_val = min(1.0, 2 * lam_min)
    s = np.full(p, s_val)
    C = 2 * np.diag(s) - np.diag(s) @ Sigma_inv @ np.diag(s)
    # Ensure PSD
    C_psd = (C + C.T) / 2
    vals, vecs = np.linalg.eigh(C_psd)
    vals = np.clip(vals, 1e-8, None)
    L = vecs @ np.diag(np.sqrt(vals))
    Z = rng.normal(0, 1, (n, p))
    X_tilde = X @ (np.eye(p) - Sigma_inv @ np.diag(s)) + Z @ L.T
    return X_tilde


def knockoff_filter(W, q=0.10):
    """Return selection indices satisfying FDR <= q + eps."""
    abs_W = np.abs(W)
    thresholds = np.sort(np.unique(abs_W[abs_W > 0]))
    tau_star = None
    for tau in thresholds:
        neg = int((W <= -tau).sum())
        pos = int((W >= tau).sum())
        if pos > 0 and (1 + neg) / pos <= q:
            tau_star = tau; break
    if tau_star is None:
        return np.zeros_like(W, dtype=bool), None
    return W >= tau_star, tau_star


if __name__ == "__main__":
    print("=== Model-X knockoffs (Candes-Fan-Janson-Lv 2018) ===\n")
    rng = np.random.default_rng(0)
    n, p = 300, 40
    n_trials = 50
    # Need >= 1/q signals for the knockoff filter to select anything at level q.
    beta_true = np.zeros(p)
    signal_idx = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]
    beta_true[signal_idx] = 2.5
    true_supp = set(np.where(np.abs(beta_true) > 0)[0])

    fdps, powers = [], []
    for trial in range(n_trials):
        X = rng.normal(0, 1, (n, p))
        y = X @ beta_true + rng.normal(0, 1, n)
        X_tilde = equi_gaussian_knockoffs(X, seed=trial)
        X_cat = np.hstack([X, X_tilde])
        b = _lasso_coefs(X_cat, y, lam=0.03)
        Z = np.abs(b[:p])
        Z_tilde = np.abs(b[p:])
        W = Z - Z_tilde
        sel, tau = knockoff_filter(W, q=0.15)
        sel_idx = set(np.where(sel)[0])
        fp = len(sel_idx - true_supp)
        tp = len(sel_idx & true_supp)
        fdp = fp / max(len(sel_idx), 1)
        power = tp / len(true_supp)
        fdps.append(fdp); powers.append(power)

    print(f"  target FDR q = 0.15, over {n_trials} trials  (12 true signals)")
    print(f"  empirical FDR = {np.mean(fdps):.3f}   (should be <= q + noise)")
    print(f"  empirical power (true-positive rate) = {np.mean(powers):.3f}")
    print("\n--- library cross-check (knockoff R package; python knockpy) ---")
