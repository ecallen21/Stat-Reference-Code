"""Oaxaca-Blinder decomposition (Reference Sec 35.21).

Blinder (1973), Oaxaca (1973).  Decompose the mean outcome gap between
two groups (A, B) into an EXPLAINED (endowments) + UNEXPLAINED
(coefficients + interaction) piece.

Threefold decomposition (from group A's perspective):
  y_A_bar - y_B_bar  =  (X_A_bar - X_B_bar) beta_B          <- endowments
                        + X_B_bar (beta_A - beta_B)          <- coefficients
                        + (X_A_bar - X_B_bar) (beta_A - beta_B) <- interaction

Twofold with reference beta*:
  gap  =  (X_A_bar - X_B_bar) beta*                       <- explained
        + X_A_bar (beta_A - beta*) + X_B_bar (beta* - beta_B)  <- unexplained
  Common choice beta* = pooled OLS coefficient.

Widely used for wage-gap studies (gender, race, region).

Here we implement both decompositions on synthetic wage-gap data.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def oaxaca_blinder(X_A, y_A, X_B, y_B):
    beta_A = ols(X_A, y_A); beta_B = ols(X_B, y_B)
    X_A_bar = X_A.mean(axis=0); X_B_bar = X_B.mean(axis=0)
    d_X = X_A_bar - X_B_bar
    d_beta = beta_A - beta_B
    # Threefold
    endow = float(d_X @ beta_B)
    coef = float(X_B_bar @ d_beta)
    inter = float(d_X @ d_beta)
    # Twofold with pooled beta
    X_pool = np.vstack([X_A, X_B]); y_pool = np.concatenate([y_A, y_B])
    beta_star = ols(X_pool, y_pool)
    explained = float(d_X @ beta_star)
    unexplained = float(X_A_bar @ (beta_A - beta_star)
                          + X_B_bar @ (beta_star - beta_B))
    return {"gap": float(y_A.mean() - y_B.mean()),
             "threefold": {"endowments": endow, "coefficients": coef, "interaction": inter},
             "twofold": {"explained": explained, "unexplained": unexplained},
             "beta_A": beta_A, "beta_B": beta_B, "beta_star": beta_star}


if __name__ == "__main__":
    print("=== Oaxaca-Blinder decomposition ===\n")
    rng = np.random.default_rng(0)
    # Group A (women): higher education, lower coefficient (discrimination)
    n_A = 500; n_B = 500
    educ_A = rng.normal(14, 2, n_A)
    exp_A = rng.normal(10, 3, n_A)
    X_A = np.stack([np.ones(n_A), educ_A, exp_A], axis=1)
    beta_A_true = np.array([0.5, 0.06, 0.03])          # 6% return per year of education
    y_A = X_A @ beta_A_true + rng.normal(0, 0.2, n_A)

    educ_B = rng.normal(13, 2, n_B)                     # slightly less education
    exp_B = rng.normal(12, 3, n_B)                      # slightly more experience
    X_B = np.stack([np.ones(n_B), educ_B, exp_B], axis=1)
    beta_B_true = np.array([0.6, 0.10, 0.03])          # 10% return per year (bigger)
    y_B = X_B @ beta_B_true + rng.normal(0, 0.2, n_B)

    r = oaxaca_blinder(X_A, y_A, X_B, y_B)
    print(f"  observed log-wage gap (A - B) = {r['gap']:.4f}")
    print(f"\n  Threefold decomposition (A vs B):")
    for k, v in r['threefold'].items():
        print(f"    {k:>13}  = {v:>+.4f}")
    print(f"    total          = {sum(r['threefold'].values()):.4f}")
    print(f"\n  Twofold decomposition (pooled-beta reference):")
    for k, v in r['twofold'].items():
        print(f"    {k:>13}  = {v:>+.4f}")
    print(f"    total          = {sum(r['twofold'].values()):.4f}\n")

    print("--- library cross-check (R oaxaca; Python oaxaca-blinder pip pkg) ---")
