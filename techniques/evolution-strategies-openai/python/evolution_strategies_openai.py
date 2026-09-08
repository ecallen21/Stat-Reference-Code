"""Evolution Strategies (OpenAI, Sec 47.147).

Salimans, Ho, Chen, Sidor & Sutskever 2017 'Evolution Strategies as
a Scalable Alternative to Reinforcement Learning'. Antithetic
finite-difference gradient estimator on the smoothed objective
F(theta) = E_{eps~N(0,I)}[f(theta + sigma eps)]:

    grad_theta F ~ (1 / (n sigma)) * sum_i f(theta + sigma eps_i) eps_i

with reward-rank normalization for scale invariance. Highly
parallelisable (workers only need scalar returns + seeds).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rank_scores(f):
    """Center-normalise ranks to [-0.5, 0.5]; a scale-free surrogate for f."""
    n = len(f)
    order = np.argsort(np.argsort(f))       # ranks in [0, n-1]
    return order / (n - 1) - 0.5            # centered in [-0.5, 0.5]


def openai_es(f, x0, sigma=0.1, lr=0.03, pop=40, n_iter=200, seed=0):
    """OpenAI ES with antithetic sampling + rank shaping (minimises f)."""
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    d = len(x)
    hist = [float(f(x))]
    for it in range(n_iter):
        # Antithetic pairs: reduce variance by using +eps and -eps
        eps = rng.normal(size=(pop // 2, d))
        eps = np.concatenate([eps, -eps], axis=0)
        fits = np.array([f(x + sigma * e) for e in eps])
        r = rank_scores(-fits)              # invert: we minimise, so bigger rank <- smaller f
        grad = (r[:, None] * eps).sum(0) / (pop * sigma)
        x = x + lr * grad                   # ascend rank-shaped surrogate
        hist.append(float(f(x)))
    return {"x": x, "f": float(f(x)), "history": hist}


if __name__ == "__main__":
    print("=== Evolution Strategies (Salimans et al 2017, OpenAI) ===\n")

    # Test 1: quadratic bowl in 10-D
    def bowl(x):
        return float(np.sum(x ** 2))

    x0 = np.ones(10) * 3.0
    r = openai_es(bowl, x0, sigma=0.5, lr=0.5, pop=40, n_iter=300, seed=0)
    print(f"  10-D bowl:  f(x0) = {bowl(x0):.3f}   f_final = {r['f']:.6f}   "
          f"||x||_inf = {np.max(np.abs(r['x'])):.4f}")

    # Test 2: 5-D noisy step-function (piecewise-constant, no gradient anywhere)
    def step5(x):
        return float(np.sum(np.floor(x) ** 2))

    rng = np.random.default_rng(0)
    x0 = rng.uniform(2, 4, size=5)
    for iters in [50, 200, 500]:
        r = openai_es(step5, x0, sigma=0.5, lr=0.2, pop=40, n_iter=iters, seed=0)
        print(f"  5-D step function iters={iters:3d}   f_best = {r['f']:.2f}   "
              f"floor(x) = {np.floor(r['x']).astype(int)}")

    print("\n  ES doesn't need gradients — only forward evaluations of f,")
    print("  which is why it scales to piecewise-constant / non-differentiable")
    print("  landscapes (RL policies, black-box simulators, etc).")

    print("\n--- library cross-check (evosax / cma-es Python; nevergrad) ---")
