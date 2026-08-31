"""Fused LASSO / Total-Variation regularisation (Reference Sec 32.13).

Tibshirani, Saunders, Rosset, Zhu & Knight (2005) 'Sparsity and
smoothness via the fused LASSO.'

Adds a penalty on ADJACENT differences of coefficients:

  min_beta   0.5 || y - X beta ||^2
              + lambda_1 * sum_j |beta_j|
              + lambda_2 * sum_j |beta_j - beta_{j-1}|

For 1-D signal denoising (X = I): FUSED LASSO reduces to TOTAL-VARIATION
(TV) denoising -> preserves piecewise-constant structure while removing
noise. Classic in changepoint detection, image denoising, CGH arrays,
copy-number variation.

Here we implement TV denoising via the compact 'Taut String' /
projected-gradient approximation, and apply to a piecewise-constant
signal with Gaussian noise.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def tv_denoise(y, lam=0.5, max_iter=500, lr=0.5):
    """Chambolle 2004 dual projection for 1-D TV denoising."""
    n = len(y)
    if n < 2: return y.copy()
    p = np.zeros(n - 1)
    for _ in range(max_iter):
        # Primal: x = y - D' p, where D is the finite-diff operator (n-1) x n.
        Dt_p = np.zeros(n)
        Dt_p[:-1] += -p
        Dt_p[1:] += p
        x = y - Dt_p
        # Dual gradient step
        Dx = np.diff(x)
        p = p + lr / 4 * Dx
        # Projection to [-lam, lam]
        p = np.clip(p, -lam, lam)
    Dt_p = np.zeros(n)
    Dt_p[:-1] += -p
    Dt_p[1:] += p
    return y - Dt_p


if __name__ == "__main__":
    print("=== Fused LASSO / Total-Variation denoising (Tibshirani 2005) ===\n")
    rng = np.random.default_rng(0)
    n = 120
    # Piecewise-constant signal: three plateaux.
    truth = np.concatenate([np.ones(40) * 0.5,
                              np.ones(40) * -0.4,
                              np.ones(40) * 1.5])
    y = truth + rng.normal(0, 0.3, n)

    for lam in (0.05, 0.15, 0.4):
        x_hat = tv_denoise(y, lam=lam, max_iter=500)
        mse_raw = float(np.mean((y - truth) ** 2))
        mse_tv = float(np.mean((x_hat - truth) ** 2))
        # Count 'jumps' (change-points) with tolerance
        n_jumps = int((np.abs(np.diff(x_hat)) > 0.20).sum())
        print(f"  lambda={lam:.2f}   raw_MSE={mse_raw:.4f}   TV_MSE={mse_tv:.4f}"
              f"   estimated changepoints={n_jumps}   (true=2)")

    print("\n  TV denoising recovers plateaux + a small number of changepoints.\n")
    print("--- library cross-check (skimage.restoration.denoise_tv_chambolle;"
          " R genlasso::fusedlasso1d; ruptures for changepoint) ---")
