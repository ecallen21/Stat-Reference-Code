"""Minimum Description Length (MDL) (Reference Sec 34.8).

Rissanen (1978, 1996) 'Modeling by shortest data description.'

MODEL SELECTION principle: pick the model that minimises total
description length:

  L_total(y, M)  =  L_model(M)  +  L_data(y | M).

Two-part MDL:  L_model = parameter code length ~ (k/2) log n;
                L_data  = -log p(y | theta_hat).
  -> BIC = -2 log L + k log n.

Normalised Maximum Likelihood (NML) code (Rissanen 1996) is a
parameter-free improvement:
  L_NML(y | M) = -log( p(y | theta_hat(y)) / C(M) )
  with C(M) = sum_y p(y | theta_hat(y)) (normalising constant, model-complexity term).

Here we illustrate two-part MDL on nested polynomial regressions +
show the connection to BIC.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fit_ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float(resid @ resid / len(y))
    return beta, sigma2


def gaussian_neg_log_lik(X, y, beta, sigma2):
    n = len(y); resid = y - X @ beta
    return 0.5 * n * np.log(2 * np.pi * sigma2) + (resid @ resid) / (2 * sigma2)


def mdl_two_part(X, y, k):
    beta, sigma2 = fit_ols(X, y)
    n = len(y)
    L_data = float(gaussian_neg_log_lik(X, y, beta, sigma2))
    L_model = 0.5 * k * np.log(n)                    # standard 2-part encoding
    return L_data + L_model, L_data, L_model


if __name__ == "__main__":
    print("=== Minimum Description Length (Rissanen 1978) ===\n")
    rng = np.random.default_rng(0)
    n = 100
    x = rng.uniform(-2, 2, n)
    Phi_true = np.stack([np.ones(n), x, x ** 2, x ** 3], axis=1)
    beta_true = np.array([0.5, 1.0, -0.5, 0.3])
    y = Phi_true @ beta_true + rng.normal(0, 0.5, n)

    print(f"  {'order':>6}  {'k':>3}  {'L_data':>10}  {'L_model':>10}  {'L_total':>10}"
          f"  {'BIC':>10}")
    for order in range(1, 8):
        cols = [x ** j for j in range(order + 1)]
        Phi = np.stack(cols, axis=1)
        k = order + 2                                # add sigma^2
        L, Ld, Lm = mdl_two_part(Phi, y, k)
        bic = 2 * Ld + k * np.log(n)                 # equivalent parameterisation
        print(f"  {order:>6}  {k:>3}  {Ld:>10.2f}  {Lm:>10.2f}  {L:>10.2f}"
              f"  {bic:>10.2f}")

    print("\n  Minimum L_total picks the model with the shortest total code.")
    print("  MDL asymptotically equivalent to BIC for regular parametric families.\n")
    print("--- library cross-check (R stats::BIC; R minMDL; Python custom + statsmodels) ---")
