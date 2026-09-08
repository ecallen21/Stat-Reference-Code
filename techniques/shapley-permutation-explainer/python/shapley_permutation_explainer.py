"""Shapley Permutation / Kernel SHAP explainer (Reference Sec 47.130).

Strumbelj & Kononenko 2010 (permutation SHAP); Lundberg & Lee 2017
'A unified approach to interpreting model predictions', NeurIPS
(Kernel SHAP). Permutation SHAP:

    phi_i^perm(x) = (1 / M) sum_{sigma}
        [ f(x_{sigma_1:i, sigma_{i+1:d}=baseline}) - f(x_{sigma_1:i-1, sigma_i:d}=baseline) ]

Kernel SHAP fits a weighted linear regression on subset-inclusion
indicators with Shapley kernel weights, giving the same phi in
expectation but faster on average.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from itertools import permutations    # exact enumeration for small d


def permutation_shap(model, x, baseline, n_perms=None, rng=None):
    """Sample-based Shapley values via permutations."""
    rng = rng or np.random.default_rng(0)
    d = len(x)
    if n_perms is None:
        perms = list(permutations(range(d)))
    else:
        perms = [rng.permutation(d) for _ in range(n_perms)]
    phi = np.zeros(d)
    for sigma in perms:
        x_curr = baseline.copy()
        f_prev = float(model(x_curr))
        for i in sigma:
            x_curr[i] = x[i]
            f_now = float(model(x_curr))
            phi[i] += f_now - f_prev
            f_prev = f_now
    return phi / len(perms)


def kernel_shap(model, x, baseline, n_samples=200, rng=None):
    """Kernel SHAP: fit weighted linear regression on random subsets."""
    rng = rng or np.random.default_rng(0)
    d = len(x)
    subsets = []; vals = []; weights = []
    from math import comb
    def kernel(z_sum):
        if z_sum == 0 or z_sum == d: return 1e6
        return (d - 1) / (comb(d, z_sum) * z_sum * (d - z_sum))
    for _ in range(n_samples):
        S = rng.integers(0, 2, size=d)
        z_sum = int(S.sum())
        x_s = np.where(S == 1, x, baseline)
        subsets.append(S); vals.append(float(model(x_s))); weights.append(kernel(z_sum))
    Z = np.array(subsets, dtype=float)
    v = np.array(vals); w = np.array(weights)
    # Solve weighted LS with intercept
    X_ = np.column_stack([np.ones(len(Z)), Z])
    W = np.diag(w)
    beta = np.linalg.solve(X_.T @ W @ X_, X_.T @ W @ v)
    return beta[1:]


if __name__ == "__main__":
    print("=== Shapley Permutation / Kernel SHAP (Strumbelj-Kononenko 2010; Lundberg-Lee 2017) ===\n")
    rng = np.random.default_rng(0)
    # Toy model: f(x) = x_0 + 2 x_1 - x_2 + 0.5 x_1 x_3
    def f(x): return x[0] + 2 * x[1] - x[2] + 0.5 * x[1] * x[3]

    baseline = np.zeros(4)
    x = np.array([1.0, 1.0, 1.0, 1.0])

    print(f"  f(baseline) = {f(baseline):.3f}")
    print(f"  f(x)         = {f(x):.3f}  ->  Shapley values should sum to {f(x)-f(baseline):.3f}\n")

    perm_exact = permutation_shap(f, x, baseline)         # all 24 perms
    perm_mc = permutation_shap(f, x, baseline, n_perms=200, rng=rng)
    kshap = kernel_shap(f, x, baseline, n_samples=800, rng=rng)
    print(f"  Feature |  perm-exact  perm-MC(200)  Kernel-SHAP(800)")
    print(f"  --------+-------------------------------------")
    for i in range(4):
        print(f"    x_{i}  |    {perm_exact[i]:+.3f}       {perm_mc[i]:+.3f}         {kshap[i]:+.3f}")
    print(f"  --------+-------------------------------------")
    print(f"    sum  |    {perm_exact.sum():+.3f}       {perm_mc.sum():+.3f}         {kshap.sum():+.3f}")

    print("\n--- library cross-check (shap Python; iml / DALEX / fastshap R) ---")
