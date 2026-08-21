"""SHAP values for model explanation (Reference §21.x extra; Lundberg-Lee 2017).

Shapley values from cooperative game theory: for feature i, the SHAP value
phi_i(x) is the average marginal contribution over all coalitions:

    phi_i(x) = sum_{S ⊆ N \\ {i}} |S|! (n - |S| - 1)! / n!
               * ( f(x_{S ∪ {i}}) - f(x_S) )

Missing features in the coalition are integrated out using a background
distribution (row sample from the training data).

Enumerating all 2^n coalitions is O(2^n); we do this exactly for small p
(≤ ~12).  For larger p, use Kernel SHAP (Lundberg-Lee 2017) or Tree SHAP
(Lundberg 2020, O(TLD^2)) via the `shap` package.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math (factorial, log)

from itertools import combinations    # stdlib: subset enumeration

import numpy as np    # numerical arrays + linear algebra


def _shap_weight(s: int, n: int) -> float:
    return math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)


def exact_shap(predict, x, background) -> np.ndarray:
    """Exact Shapley values by enumerating all 2^p coalitions.

    predict:   callable that takes an (m, p) array and returns an (m,) prediction.
    x:         (p,) instance to explain.
    background: (m, p) rows used to marginalize missing features.
    """
    x = np.asarray(x, dtype=float); background = np.asarray(background, dtype=float)
    p = len(x); n_back = len(background); phi = np.zeros(p)
    features = list(range(p))
    # cache expected prediction under each coalition
    def _expected(S):
        X = np.tile(x, (n_back, 1))                       # copy of x per background row
        for j in features:
            if j not in S:
                X[:, j] = background[:, j]                # marginalise j via background
        return float(predict(X).mean())
    # enumerate coalitions
    for i in features:
        others = [j for j in features if j != i]
        for k in range(len(others) + 1):
            for S in combinations(others, k):
                Sset = set(S)
                delta = _expected(Sset | {i}) - _expected(Sset)
                phi[i] += _shap_weight(k, p) * delta
    return phi


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Linear model: SHAP values should equal (x_i - E[x_i]) * beta_i exactly
    n = 300; p = 4
    beta = np.array([0.5, -1.2, 0.0, 2.0])
    X = rng.normal(size=(n, p))
    y = X @ beta + rng.normal(scale=0.3, size=n)

    from sklearn.linear_model import LinearRegression
    m = LinearRegression().fit(X, y)
    predict = lambda A: m.predict(A)

    background = X[:100]
    x_star = np.array([1.0, -0.5, 1.5, 0.3])
    phi = exact_shap(predict, x_star, background)

    # analytical: for linear f, phi_i = beta_i (x_i - E[x_i])
    E_bg = background.mean(axis=0)
    phi_analytic = m.coef_ * (x_star - E_bg)

    print("=== Exact SHAP for a linear model (should equal beta_i(x_i - mean_i)) ===")
    print(f"  {'j':>2} {'exact':>10} {'analytic':>10}   diff")
    for j in range(p):
        print(f"  {j:>2} {phi[j]:>10.4f} {phi_analytic[j]:>10.4f}   "
              f"{abs(phi[j] - phi_analytic[j]):.2e}")
    # efficiency check: sum(phi) = f(x) - E[f(X)]
    f_x = float(m.predict(x_star.reshape(1, -1))[0])
    E_f = float(m.predict(background).mean())
    print(f"\n  sum(phi)    = {phi.sum():.4f}")
    print(f"  f(x) - E[f] = {f_x - E_f:.4f}")

    print("\n--- library cross-check (shap.KernelExplainer) ---")
    try:
        import shap
        ke = shap.KernelExplainer(predict, background)
        sv = ke.shap_values(x_star, nsamples=200, silent=True)
        print("  shap.KernelExplainer values: " + "  ".join(f"{v:.4f}" for v in sv))
    except ImportError:
        print("  (shap not installed; from-scratch numbers only)")
