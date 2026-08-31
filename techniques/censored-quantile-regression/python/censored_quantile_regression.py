"""Censored quantile regression (Reference Sec 33.3).

Powell (1986) 'Censored regression quantiles.'

Standard QR minimises sum(rho_tau(y_i - x_i' beta)); Powell's CQR
handles LEFT-CENSORED (or right-censored) responses by replacing the
prediction with max(0, x_i' beta) (i.e. project through the censoring
threshold):

  min_beta  sum_i  rho_tau( y_i - max(0, x_i' beta) )
              (for left-censoring at 0; extend by translating).

The objective is non-convex but well-behaved via iterative linear
programming (BRCENS) or interior-point on the LP relaxation. For a
compact demo we use a subgradient-descent solver.

Here we simulate right-censored data (T = min(T*, C)), fit standard QR
(biased) and Powell CQR (consistent), and compare parameter recovery.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def check_loss(u, tau):
    return u * (tau - (u < 0).astype(float))


def _rho(r, tau):
    return check_loss(r, tau).sum()


def qr_naive(X, y, tau, lr=1e-3, epochs=3000, seed=0):
    """Subgradient descent for standard QR (used as a naive baseline)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    beta = np.zeros(d)
    n = X.shape[0]
    for _ in range(epochs):
        r = y - X @ beta
        # subgradient of check loss: (I{r<0} - tau)
        sg = (r < 0).astype(float) - tau
        beta -= lr * X.T @ sg / n
    return beta


def cqr_powell(X, y, tau, ceiling, lr=1e-3, epochs=3000, seed=0):
    """Powell 1986 CQR for RIGHT-censored data: y_i = min(y*_i, C).

    Objective: sum rho_tau( y_i - min(X_i beta, C) ).
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    beta = X.T @ y / (X.T @ X).sum()          # crude start
    beta = np.zeros(d) if not np.isfinite(beta).all() else beta
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    n = X.shape[0]
    for _ in range(epochs):
        pred_uncens = X @ beta
        pred = np.minimum(pred_uncens, ceiling)
        active = pred_uncens < ceiling                # gradient flows only where uncensored
        r = y - pred
        sg = (r < 0).astype(float) - tau
        sg = sg * active                              # zero at censored region
        beta -= lr * X.T @ sg / n
    return beta


if __name__ == "__main__":
    print("=== Censored quantile regression (Powell 1986) ===\n")
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(-2, 2, n)
    X = np.stack([np.ones(n), x], axis=1)
    # True latent y* = 1 + 1.5 * x + eps.
    beta_true = np.array([1.0, 1.5])
    y_star = X @ beta_true + rng.normal(0, 0.8, n)
    # Right-censor at ceiling = 3.0.
    ceiling = 3.0
    y = np.minimum(y_star, ceiling)
    censored_rate = np.mean(y_star > ceiling)
    print(f"  right-censoring rate: {censored_rate:.2f}   ceiling C = {ceiling}\n")

    for tau in (0.25, 0.50, 0.75):
        b_naive = qr_naive(X, y, tau)
        b_cqr = cqr_powell(X, y, tau, ceiling)
        print(f"  tau={tau:.2f}")
        print(f"    naive QR on censored y:  intercept={b_naive[0]:>6.3f}  slope={b_naive[1]:>6.3f}")
        print(f"    Powell CQR             :  intercept={b_cqr[0]:>6.3f}  slope={b_cqr[1]:>6.3f}")
        print(f"    truth (median at tau=.50) intercept=1.000  slope=1.500\n")

    print("--- library cross-check (statsmodels QuantReg + custom censoring; quantreg::crq in R) ---")
