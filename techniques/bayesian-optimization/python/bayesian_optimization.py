"""Bayesian optimization with a Gaussian-process surrogate (Reference §14.28).

Sequential black-box optimization when each function evaluation is expensive
(hyperparameter tuning, physical experiments, drug screens).  Model the
unknown function f(x) with a Gaussian process, pick the next x by
maximizing an ACQUISITION function that trades exploration vs exploitation.

Steps
    1. Fit GP posterior to observations (x_i, y_i).
    2. Compute acquisition alpha(x) over a candidate grid.
    3. Query the true f at argmax alpha(x); append to data; repeat.

Acquisition functions
    - Probability of Improvement (PI)  Kushner 1964.
    - Expected Improvement (EI)  Mockus 1978:
        EI(x) = (mu(x) - f_best - xi) Phi(z) + sigma(x) phi(z)
        z = (mu(x) - f_best - xi) / sigma(x)
    - Upper Confidence Bound (UCB, Srinivas 2010): mu(x) + kappa sigma(x)

The demo below uses EI with a squared-exponential kernel.  For production
use scikit-optimize, GPyOpt, botorch, or the R package DiceOptim.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def rbf_kernel(X, Y, length_scale: float = 1.0, sigma_f: float = 1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X ** 2, 1)[:, None] + np.sum(Y ** 2, 1)[None, :] - 2 * X @ Y.T
    return sigma_f ** 2 * np.exp(-0.5 * d2 / length_scale ** 2)


def gp_posterior(X_train, y_train, X_test, length_scale: float = 1.0,
                 sigma_n: float = 0.01) -> dict:
    """GP posterior mean and variance."""
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.atleast_2d(np.asarray(X_test, dtype=float))
    K = rbf_kernel(X_train, X_train, length_scale) + (sigma_n ** 2) * np.eye(len(X_train))
    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X_train)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    K_s = rbf_kernel(X_train, X_test, length_scale)
    mu = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    var = rbf_kernel(X_test, X_test, length_scale).diagonal() - np.sum(v ** 2, 0)
    return {"mean": mu, "variance": np.maximum(var, 1e-12)}


def expected_improvement(mu, sigma, y_best, xi: float = 0.01):
    with np.errstate(divide="ignore"):
        z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    ei[sigma < 1e-8] = 0
    return ei


def bayes_opt(f, x_grid, n_init: int = 3, n_iter: int = 15,
              length_scale: float = 1.0, xi: float = 0.01, seed: int = 0) -> dict:
    """Bayesian optimization via GP + EI on a 1-D candidate grid."""
    rng = np.random.default_rng(seed)
    x_grid = np.asarray(x_grid).reshape(-1, 1)
    idx = rng.choice(len(x_grid), size=n_init, replace=False)
    X = x_grid[idx].copy(); y = np.array([f(float(x)) for x in X.ravel()])
    trajectory = list(zip(X.ravel().tolist(), y.tolist()))
    for _ in range(n_iter):
        post = gp_posterior(X, y, x_grid, length_scale=length_scale)
        ei = expected_improvement(post["mean"], np.sqrt(post["variance"]), y.max(), xi)
        next_i = int(np.argmax(ei))
        x_new = float(x_grid[next_i, 0])
        y_new = float(f(x_new))
        X = np.vstack([X, [[x_new]]])
        y = np.append(y, y_new)
        trajectory.append((x_new, y_new))
    best_idx = int(np.argmax(y))
    return {"x_best": float(X[best_idx, 0]), "y_best": float(y[best_idx]),
            "n_evals": len(y),
            "trajectory": trajectory,
            "length_scale": length_scale, "xi": xi,
            "method": "Bayesian optimization (GP + Expected Improvement)"}


if __name__ == "__main__":
    # Multimodal 1-D target with interior optimum: maximize
    def f(x): return math.sin(3 * x) + math.exp(-(x - 1.5) ** 2)

    x_grid = np.linspace(-3, 4, 501)

    print("=== Bayesian optimization of a multimodal 1-D target ===")
    r = bayes_opt(f, x_grid, n_init=3, n_iter=15, length_scale=0.8, seed=1)
    print(f"  x_best = {r['x_best']:.3f}, f(x_best) = {r['y_best']:.4f}")
    print(f"  total evals = {r['n_evals']}")

    # Compare against random-search baseline of equal budget
    rng = np.random.default_rng(1)
    xs = rng.uniform(-3, 4, 18)
    ys = np.array([f(x) for x in xs])
    print(f"\n=== Random search (18 evals) ===")
    print(f"  x_best = {xs[np.argmax(ys)]:.3f}, f(x_best) = {ys.max():.4f}")

    # True max over the grid
    ys_grid = np.array([f(x) for x in x_grid])
    print(f"\n=== Truth (grid search) ===")
    print(f"  x_max = {x_grid[np.argmax(ys_grid)]:.3f}, f(x_max) = {ys_grid.max():.4f}")

    print("\n--- library cross-check (scikit-optimize) ---")
    try:
        from skopt import gp_minimize
        res = gp_minimize(lambda x: -f(x[0]), [(-3.0, 4.0)], n_calls=18, random_state=1)
        print(f"  skopt gp_minimize: x = {res.x[0]:.3f}, f = {-res.fun:.4f}")
    except Exception as ex:
        print(f"  (skopt unavailable: {ex})")
