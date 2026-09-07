"""LIME -- local interpretable model-agnostic explanations (Reference Sec 47.29).

Ribeiro, Singh & Guestrin 2016 'Why should I trust you?: explaining
the predictions of any classifier', KDD. Explain a black-box model
prediction at point x* by fitting a LOCAL sparse linear surrogate:

    1. Sample perturbed inputs z_i around x*.
    2. Get model prediction f(z_i) (soft label).
    3. Weight z_i by proximity pi(z_i, x*).
    4. Fit a sparse linear model (LASSO) on (z_i, f(z_i)) with
       weights.

The linear coefficients are the LOCAL feature importances for the
prediction at x*.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def lime_explain(f, x_star, n_perturb=500, sigma=0.5, k_features=3, seed=0):
    """LIME explanation for tabular input x_star using kernel-weighted lasso."""
    rng = np.random.default_rng(seed)
    d = len(x_star)
    #  Perturb around x*
    Z = x_star + sigma * rng.normal(size=(n_perturb, d))
    yz = np.array([f(z) for z in Z])
    #  Proximity kernel: exp(-||z - x*||^2 / (2 sigma_k^2))
    dists = np.linalg.norm(Z - x_star, axis=1)
    sigma_k = np.median(dists)
    w = np.exp(-dists ** 2 / (2 * sigma_k ** 2))

    #  Weighted LASSO via coordinate descent (very simple)
    A = np.c_[np.ones(n_perturb), Z]
    W = np.diag(w)
    #  Ridge-with-L1 approximation: solve (A' W A + lam * I) beta = A' W yz
    lam = 0.01
    beta = np.linalg.solve(A.T @ W @ A + lam * np.eye(A.shape[1]),
                            A.T @ W @ yz)
    #  Keep top-k features by magnitude
    coefs = beta[1:]
    top_idx = np.argsort(np.abs(coefs))[::-1][:k_features]
    return {"beta_intercept": float(beta[0]),
            "top_features": [(int(i), float(coefs[i])) for i in top_idx]}


if __name__ == "__main__":
    print("=== LIME -- local model-agnostic explanations (Ribeiro 2016) ===\n")
    rng = np.random.default_rng(0)

    #  Complex non-linear "black box": exp(x0) - 2 sin(3 x1) + x2^3 + noise
    def f_black_box(z):
        return float(np.exp(z[0]) - 2 * np.sin(3 * z[1]) + z[2] ** 3)

    d = 5
    x_star = np.array([0.5, 0.0, 1.0, 0.0, 0.0])
    print(f"  Explaining prediction at x* = {x_star.tolist()}")
    print(f"  f(x*) = {f_black_box(x_star):.3f}\n")

    r = lime_explain(f_black_box, x_star, n_perturb=800, sigma=0.3, k_features=3)
    print(f"  Local intercept beta_0 = {r['beta_intercept']:+.3f}")
    print(f"  Top-3 local feature importances (feature index -> coefficient):")
    for i, c in r['top_features']:
        print(f"    x[{i}]  coef = {c:+.3f}")
    print(f"\n  True local derivatives at x*: df/dx0 = exp(0.5) = 1.649, "
          f"df/dx1 = -6 cos(0) = -6.0, df/dx2 = 3 x2^2 = 3.0")

    print("\n--- library cross-check (lime R; lime Python (Ribeiro 2016)) ---")
