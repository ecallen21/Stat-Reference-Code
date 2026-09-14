"""Cross-Entropy method (Rubinstein 1997, 1999).

Iterative importance-sampling for rare-event simulation
and for optimisation:

    (1) sample X ~ p(theta_t)
    (2) select elite fraction (top rho)
    (3) refit theta_{t+1} to elites by MLE

The parametric family p(theta) is often Gaussian for
continuous optimisation, categorical for discrete.
"""

import numpy as np    # arrays + random


def rastrigin(x):
    A = 10
    d = x.shape[-1]
    return A * d + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=-1)


def sphere(x):
    return np.sum(x ** 2, axis=-1)


def cross_entropy(f, dim, pop_size=100, elite_frac=0.2, n_iter=40,
                  init_std=3.0, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-4, 4, size=dim)
    sigma = np.full(dim, init_std)
    n_elite = int(pop_size * elite_frac)
    best_x, best_f = None, np.inf
    hist_best = []
    for _ in range(n_iter):
        X = mu + sigma * rng.standard_normal((pop_size, dim))
        fx = f(X)
        best_idx = int(np.argmin(fx))
        if fx[best_idx] < best_f:
            best_f = float(fx[best_idx])
            best_x = X[best_idx].copy()
        elite_idx = np.argsort(fx)[:n_elite]
        elites = X[elite_idx]
        mu = elites.mean(axis=0)
        sigma = elites.std(axis=0) + 1e-3
        hist_best.append(best_f)
    return best_x, best_f, mu, hist_best


def demo():
    print("=== Cross-Entropy Method (Rubinstein 1997) ===")
    print("Sphere function d=10, global min 0 at 0")
    x, fx, mu, hist = cross_entropy(sphere, dim=10, pop_size=200, elite_frac=0.1,
                                    n_iter=30, init_std=4.0)
    print(f"  best  x  norm = {np.linalg.norm(x):.2e}")
    print(f"  best f(x)      = {fx:.2e}")
    print(f"  best-so-far    : {[f'{h:.2f}' for h in hist[::5]]}")

    print("\nRastrigin d=5 (multi-modal, global min 0 at 0)")
    x, fx, mu, hist = cross_entropy(rastrigin, dim=5, pop_size=500, elite_frac=0.1,
                                    n_iter=100, init_std=4.0)
    print(f"  best  x        = {np.round(x, 3)}")
    print(f"  best f(x)      = {fx:.3f}  (0 = global optimum)")


if __name__ == "__main__":
    demo()
