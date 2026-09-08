"""CMA-ES - Covariance Matrix Adaptation Evolution Strategy (§47.138).

Hansen & Ostermeier 2001 'Completely derandomized self-adaptation
in evolution strategies', Evol Comp 9(2). Population-based
gradient-free optimizer that adapts a multivariate-Gaussian search
distribution to the local landscape:

    x_i ~ mean + sigma * N(0, C),   i = 1..lambda
    rank by f, keep top mu
    update mean, sigma, C from weighted top-mu.

Considered state of the art for continuous black-box optimisation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def cma_es(f, x0, sigma=1.0, n_iter=100, popsize=None):
    """Compact CMA-ES. Returns best-so-far solution and history."""
    n = len(x0)
    lam = popsize or 4 + int(3 * np.log(n))
    mu = lam // 2
    w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mu_eff = 1.0 / (w ** 2).sum()

    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

    mean = np.array(x0, dtype=float)
    C = np.eye(n)
    p_sigma = np.zeros(n); p_c = np.zeros(n)
    best_x, best_f = mean.copy(), f(mean)
    history = [best_f]

    rng = np.random.default_rng(0)
    for gen in range(n_iter):
        eigvals, B = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-12, None)
        BD = B * np.sqrt(eigvals)
        pop = np.array([mean + sigma * BD @ rng.normal(size=n) for _ in range(lam)])
        fitness = np.array([f(x) for x in pop])
        order = np.argsort(fitness)
        best_offspring = pop[order[:mu]]
        # Update mean
        new_mean = w @ best_offspring
        # Evolution paths
        C_inv_sqrt = B @ np.diag(1 / np.sqrt(eigvals)) @ B.T
        p_sigma = (1 - c_sigma) * p_sigma + \
                    np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (new_mean - mean) / sigma
        p_c = (1 - c_c) * p_c + \
                np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
        # Update C (rank-1 + rank-mu)
        Y = (best_offspring - mean) / sigma
        C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu * (w[:, None, None] * Y[:, :, None] * Y[:, None, :]).sum(axis=0)
        # Update sigma
        E_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / E_n - 1))
        mean = new_mean
        if fitness[order[0]] < best_f:
            best_f = float(fitness[order[0]]); best_x = pop[order[0]].copy()
        history.append(best_f)
    return {"x": best_x, "f": best_f, "history": history}


if __name__ == "__main__":
    print("=== CMA-ES (Hansen-Ostermeier 2001) ===\n")

    # Rosenbrock function in 5D
    def rosen(x):
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    rng = np.random.default_rng(0)
    x0 = rng.uniform(-2, 2, size=5)
    print(f"  Rosenbrock 5D  starting x0 = {np.round(x0, 3)}   f(x0) = {rosen(x0):.3f}")

    for iters in [50, 200, 500]:
        r = cma_es(rosen, x0, sigma=0.5, n_iter=iters)
        print(f"  iters = {iters:3d}   f_best = {r['f']:.6f}   ||x - 1||_∞ = {np.max(np.abs(r['x'] - 1)):.4f}")

    print("\n  Optimum at x = 1 (all ones), f = 0.")

    print("\n--- library cross-check (cmaes / pycma Python; adagio / cmaesr R) ---")
