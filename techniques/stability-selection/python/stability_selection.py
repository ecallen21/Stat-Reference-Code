"""Stability selection (Reference Sec 32.6).

Meinshausen & Buhlmann (2010) 'Stability selection.'

Repeatedly fit a selection procedure (LASSO, RF importance) on random
SUBSAMPLES of size n/2; keep features whose SELECTION FREQUENCY across
subsamples exceeds a threshold pi_thr. Under mild exchangeability
conditions the expected number of false positives is bounded by

  E[V]  <=  1 / (2 pi_thr - 1)  * q^2 / p,

where q is the average number of variables selected per subsample and
p is the total feature count.

Advantages:
  * MODEL-AGNOSTIC (wraps any selection method).
  * Formal FDR-like error control without distributional assumptions.
  * Robust to hyperparameter (lambda) choice.

Here we implement stability selection with LASSO as the base selector,
apply to synthetic sparse regression, and report the stability path.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _soft(x, lam): return np.sign(x) * np.maximum(0, np.abs(x) - lam)


def lasso_selection(X, y, lam, max_iter=100):
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
    return np.abs(beta) > 1e-6


def stability_selection(X, y, lam_grid, B=100, subsample=0.5, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    n_sub = int(n * subsample)
    freq = np.zeros((len(lam_grid), d))
    for lam_i, lam in enumerate(lam_grid):
        for _ in range(B):
            idx = rng.choice(n, n_sub, replace=False)
            sel = lasso_selection(X[idx], y[idx], lam)
            freq[lam_i] += sel.astype(int)
    return freq / B


if __name__ == "__main__":
    print("=== Stability selection (Meinshausen-Buhlmann 2010) ===\n")
    rng = np.random.default_rng(0)
    n, d = 200, 30
    beta_true = np.zeros(d)
    beta_true[[0, 5, 15]] = [2.0, -1.5, 1.8]
    X = rng.normal(0, 1, (n, d))
    y = X @ beta_true + rng.normal(0, 0.5, n)

    lam_grid = [0.05, 0.10, 0.20, 0.30]
    freq = stability_selection(X, y, lam_grid, B=60, subsample=0.5, seed=0)

    max_freq = freq.max(axis=0)                # stability score per feature
    order = np.argsort(-max_freq)
    print(f"  Stability paths (rows = lambda, cols = 30 features):")
    print(f"    max frequency per feature:")
    for j in order[:10]:
        star = " <- TRUE" if beta_true[j] != 0 else ""
        print(f"      feature {j:>2}   max_freq={max_freq[j]:.2f}{star}")

    for thr in (0.60, 0.75, 0.90):
        sel = np.where(max_freq >= thr)[0]
        tp = int(sum(beta_true[sel] != 0))
        fp = int(sum(beta_true[sel] == 0))
        print(f"\n  threshold pi_thr = {thr:.2f}   selected {len(sel)} features  TP={tp}/3   FP={fp}")

    print("\n--- library cross-check (R stabs; Python stability-selection pip pkg) ---")
