"""Dirichlet Regression (Reference Sec 47.324).

Campbell & Mosimann 1987; Maier 2014 J Stat Softw. Model
COMPOSITIONAL outcomes y in the simplex (y_j > 0, sum y_j = 1)
as Dirichlet(alpha) with alpha_j = exp(x^T beta_j):

    y | X ~ Dir(alpha(X))
    log alpha_j(x) = x^T beta_j

Fit by MLE. Compositional outcomes arise in bioinformatics
(taxa proportions), budget shares, election outcomes.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def dirichlet_nll(beta_flat, X, Y, K):
    from scipy.special import gammaln
    d = X.shape[1]
    B = beta_flat.reshape(K, d)
    alpha = np.exp(X @ B.T)                                        # (n, K)
    alpha_sum = alpha.sum(axis=1)
    ll = (gammaln(alpha_sum) - gammaln(alpha).sum(axis=1)
          + ((alpha - 1) * np.log(Y + 1e-12)).sum(axis=1))
    return float(-ll.sum())


def fit_dirichlet_reg(X, Y, max_iter=200):
    """Numerical MLE via L-BFGS."""
    from scipy.optimize import minimize
    n, d = X.shape; K = Y.shape[1]
    beta0 = np.zeros(K * d)
    res = minimize(dirichlet_nll, beta0, args=(X, Y, K), method="L-BFGS-B",
                    options={"maxiter": max_iter})
    return res.x.reshape(K, d), res.fun


if __name__ == "__main__":
    print("=== Dirichlet Regression (Campbell-Mosimann 1987; Maier 2014) ===\n")
    rng = np.random.default_rng(0)

    # Simulate composition of 3 taxa driven by 2 covariates
    n = 300
    X = np.column_stack([np.ones(n), rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    true_beta = np.array([[1.0, 2.0, -0.5],
                            [0.5, -1.0, 1.5],
                            [0.0, 0.5, 0.5]])
    alpha_true = np.exp(X @ true_beta.T)
    Y = np.array([rng.dirichlet(a) for a in alpha_true])

    print(f"  n = {n}, K = 3 taxa, d = 3 covariates (intercept + 2)")
    print(f"  Simulated compositions sum to 1 exactly: max abs error = "
          f"{np.abs(Y.sum(axis=1) - 1).max():.2e}\n")

    beta_hat, nll = fit_dirichlet_reg(X, Y, max_iter=200)
    print(f"  True beta:")
    print(f"{np.array2string(true_beta, precision=2)}")
    print(f"  Estimated beta:")
    print(f"{np.array2string(beta_hat, precision=2)}")
    print(f"\n  Final negative log-likelihood: {nll:.2f}")

    # Mean composition on a new x
    x_new = np.array([1.0, 0.5, 0.5])
    alpha_new = np.exp(x_new @ beta_hat.T)
    mean_comp = alpha_new / alpha_new.sum()
    print(f"\n  Predicted mean composition at x=(1, 0.5, 0.5):")
    print(f"    taxa 1 = {mean_comp[0]:.3f}, taxa 2 = {mean_comp[1]:.3f}, taxa 3 = {mean_comp[2]:.3f}")

    print("\n--- library cross-check (DirichletReg R; dirichlet Python; brms family=dirichlet) ---")
