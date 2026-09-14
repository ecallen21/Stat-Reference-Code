"""Genetic Matching (Reference Sec 47.327).

Diamond & Sekhon 2013 REStat. Search over a WEIGHT matrix W in
the (generalised) Mahalanobis distance:

    d(x_i, x_j; W) = sqrt((x_i - x_j)^T W (x_i - x_j))

using a GENETIC ALGORITHM to MINIMISE the worst-case KS statistic
or SMD across covariates AFTER matching. Combines propensity
scores with covariates for better balance.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def nn_match_weighted(X, treat, W):
    """1:1 NN match under d(x_i, x_j) = (x_i - x_j)^T W (x_i - x_j)."""
    treated = np.where(treat == 1)[0]; control = np.where(treat == 0)[0]
    used = set(); pairs = []
    for i in treated:
        best_d = np.inf; best_j = None
        for j in control:
            if j in used: continue
            diff = X[i] - X[j]
            d = diff @ W @ diff
            if d < best_d: best_d = d; best_j = j
        if best_j is not None:
            pairs.append((int(i), int(best_j))); used.add(best_j)
    return pairs


def max_abs_smd(X, treat, pairs):
    tt = X[[p[0] for p in pairs]]; cc = X[[p[1] for p in pairs]]
    smd = []
    for k in range(X.shape[1]):
        m_t = tt[:, k].mean(); s_t = tt[:, k].std()
        m_c = cc[:, k].mean(); s_c = cc[:, k].std()
        smd.append(abs(m_t - m_c) / (np.sqrt((s_t ** 2 + s_c ** 2) / 2) + 1e-9))
    return max(smd), smd


def genetic_match(X, treat, n_gen=20, pop_size=20, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    d = X.shape[1]
    S_inv = np.linalg.inv(np.cov(X, rowvar=False) + 1e-6 * np.eye(d))
    # Population of diagonal weight vectors (positive scaling of Mahalanobis)
    pop = rng.uniform(0.5, 2.0, size=(pop_size, d))
    for gen in range(n_gen):
        scored = []
        for w in pop:
            W = np.diag(w) @ S_inv @ np.diag(w)
            pairs = nn_match_weighted(X, treat, W)
            worst, _ = max_abs_smd(X, treat, pairs)
            scored.append((worst, w))
        scored.sort(key=lambda z: z[0])
        elites = [s[1] for s in scored[:pop_size // 4]]
        # Crossover + mutation
        offspring = []
        while len(offspring) < pop_size - len(elites):
            a, b = rng.choice(len(elites), 2, replace=True)
            child = np.where(rng.uniform(size=d) < 0.5, elites[a], elites[b])
            child = child * (1 + 0.1 * rng.standard_normal(d))
            child = np.clip(child, 0.1, 5.0)
            offspring.append(child)
        pop = np.vstack([np.array(elites), np.array(offspring)])
    # Return best
    best_w = scored[0][1]
    W_best = np.diag(best_w) @ S_inv @ np.diag(best_w)
    pairs = nn_match_weighted(X, treat, W_best)
    return best_w, pairs


if __name__ == "__main__":
    print("=== Genetic Matching (Diamond & Sekhon 2013 REStat) ===\n")
    rng = np.random.default_rng(0)

    n = 300
    X0 = rng.multivariate_normal([0, 0, 0], np.eye(3), n // 2)
    X1 = rng.multivariate_normal([0.6, 0.3, -0.4], np.eye(3), n // 2)
    X = np.vstack([X0, X1])
    treat = np.array([0] * (n // 2) + [1] * (n // 2))

    # Plain Mahalanobis baseline
    S_inv = np.linalg.inv(np.cov(X, rowvar=False))
    pairs_mahal = nn_match_weighted(X, treat, S_inv)
    worst_mahal, smd_mahal = max_abs_smd(X, treat, pairs_mahal)

    # Genetic matching
    best_w, pairs_gen = genetic_match(X, treat, n_gen=10, pop_size=16, rng=rng)
    worst_gen, smd_gen = max_abs_smd(X, treat, pairs_gen)

    print(f"  n = {n}, 3 covariates with unequal shifts across treatment groups")
    print(f"\n  Mahalanobis matching (baseline):")
    print(f"    max |SMD| = {worst_mahal:.3f}   per-cov SMD = {np.round(smd_mahal, 3).tolist()}")
    print(f"\n  Genetic matching (10 gens, pop 16):")
    print(f"    max |SMD| = {worst_gen:.3f}   per-cov SMD = {np.round(smd_gen, 3).tolist()}")
    print(f"    Best weights: {np.round(best_w, 3).tolist()}")
    print(f"\n  Genetic search TUNES the weight matrix to minimise the worst-case imbalance.")

    print("\n--- library cross-check (Matching::GenMatch R; MatchIt::matchit(method='genetic')) ---")
