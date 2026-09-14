"""MM - Majorization-Minimization (Reference Sec 47.322).

Hunter & Lange 2004 Am Stat. Instead of minimising f(x)
directly, iteratively minimise a SURROGATE g(x | x_k) that
MAJORIZES f: g >= f and g(x_k | x_k) = f(x_k). Then

    x_{k+1} = argmin_x g(x | x_k)

guarantees f(x_{k+1}) <= f(x_k). EM algorithm, IRLS, and many
convex solvers are all MM. Choosing a good surrogate makes
each step easier than the original.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def logsumexp(x): m = x.max(); return m + np.log(np.exp(x - m).sum())


def mm_median(y, n_iter=50):
    """Weiszfeld-style MM for L1 minimisation:  argmin sum |x - y_i|.
    Majoriser 0.5 * sum (x - y_i)^2 / |x_k - y_i| yields a weighted-least-squares
    step, which reduces to weighted mean."""
    x = float(np.mean(y))
    for _ in range(n_iter):
        w = 1.0 / (np.abs(x - y) + 1e-6)
        x = float(np.sum(w * y) / np.sum(w))
    return x


def mm_lasso(X, y, lam=0.1, n_iter=200):
    """MM for LASSO: bound |beta_j| by 0.5(beta_j^2/|beta_k_j| + |beta_k_j|),
    yielding a ridge subproblem each iteration."""
    n, d = X.shape
    beta = np.zeros(d); losses = []
    for _ in range(n_iter):
        w = 1.0 / (np.abs(beta) + 1e-4)
        # Ridge: (X^T X + lam * diag(w)) beta = X^T y
        A = X.T @ X + lam * np.diag(w) * n
        beta = np.linalg.solve(A, X.T @ y)
        # Soft threshold for numerical stability
        beta = np.sign(beta) * np.maximum(np.abs(beta) - 1e-4, 0)
        losses.append(0.5 * np.mean((X @ beta - y) ** 2) + lam * np.abs(beta).sum())
    return beta, losses


if __name__ == "__main__":
    print("=== MM - Majorization-Minimization (Hunter & Lange 2004) ===\n")
    rng = np.random.default_rng(0)

    # Example 1: L1 median via MM (Weiszfeld)
    y = rng.standard_normal(50); y[10] += 20; y[30] -= 25          # outliers
    x_mm = mm_median(y)
    x_median = float(np.median(y))
    print(f"  L1 median via MM: {x_mm:.4f}")
    print(f"  Sample median:    {x_median:.4f}")
    print(f"  Sample mean (for contrast): {y.mean():.4f}")
    print(f"  (L1 median is much less sensitive to outliers than the mean.)\n")

    # Example 2: LASSO via MM
    n, d = 200, 30
    X = rng.standard_normal((n, d))
    beta_true = np.zeros(d); beta_true[:5] = [3, -2, 1.5, -0.5, 2]
    yr = X @ beta_true + rng.normal(0, 0.5, n)

    beta_mm, losses = mm_lasso(X, yr, lam=0.05, n_iter=50)
    nz = (np.abs(beta_mm) > 1e-2).sum()
    print(f"  LASSO via MM (d = {d}, 5 true nonzeros): recovered {nz} nonzeros")
    print(f"  Objective trajectory:")
    for it in [0, 3, 10, 25, 49]:
        print(f"    iter {it:>3}   objective = {losses[it]:.4f}")

    print(f"\n  MM is a UMBRELLA: EM, IRLS, and ISTA are all instances.")

    print("\n--- library cross-check (see EM in mixture models; IRLS in GLM fitters) ---")
