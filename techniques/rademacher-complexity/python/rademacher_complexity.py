"""Rademacher complexity (Reference Sec 46.10).

Bartlett & Mendelson 2002. A distribution-dependent measure of a
function class F's ability to fit random labels; drives finite-sample
generalisation bounds in learning theory.

    Empirical Rademacher complexity:
        R_hat_n(F) = E_sigma [ sup_{f in F} 1/n sum_i sigma_i * f(x_i) ]

    where sigma_i are iid uniform {-1, +1} 'Rademacher noise'.

Key theorem (Bartlett-Mendelson): with probability >= 1 - delta over
the sample, for every f in F:

    E[L(f)] <= L_hat_n(f) + 2 * R_hat_n(F) + 3 * sqrt(log(2/delta) / (2n))

The 2 * R_hat_n(F) term is the 'complexity price' for hypothesis
search. Small R_hat_n => better generalisation.

We estimate R_hat_n for two classes:
    * Linear predictors with bounded norm    (F_lin = {<w, x> : ||w|| <= B})
    * VC-limited class of depth-D decision stumps
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rademacher_linear_ball(X, B=1.0, n_draws=200, seed=0):
    """R_hat for F = { <w, x> : ||w||_2 <= B }.

    Closed form for the supremum:
        sup_{||w|| <= B} (1/n) sum_i sigma_i <w, x_i>
        = B / n * || sum_i sigma_i * x_i ||_2
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    vals = []
    for _ in range(n_draws):
        sigma = rng.choice([-1, 1], size=n)
        s = sigma @ X                       # p-vector
        vals.append(B / n * np.linalg.norm(s))
    return float(np.mean(vals))


def rademacher_stumps(X, n_draws=200, seed=0):
    """R_hat for depth-1 decision stumps h(x) = sign(x[j] - t) in {-1, +1}.

    For each sigma, choose j, t maximizing (1/n) * sum sigma_i * sign(x_ij - t).
    We search cuts among midpoints of sorted x[:, j].
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    best_over_draws = []
    for _ in range(n_draws):
        sigma = rng.choice([-1, 1], size=n)
        max_score = -np.inf
        for j in range(p):
            xj = X[:, j]
            order = np.argsort(xj)
            xs = xj[order]; ss = sigma[order]
            #  Two possible sign choices at each cut.
            #  score for h = +1 if x > t: (1/n) sum sigma * sign(x - t).
            #  Vectorised: for each cut position k in [0, n),
            #  left n_l = k units get sign -1, right n_r = n - k get +1
            cum = np.cumsum(ss)
            total = cum[-1]
            #  score(cut at k) = ((total - cum[k-1]) - cum[k-1]) / n
            #                  = (total - 2*cum[k-1]) / n
            #  For k in 0..n, cum[-1] treated as 0.
            cprev = np.r_[0, cum[:-1]]
            scores = (total - 2 * cprev) / n
            #  Also allow flipping sign of the stump
            best = np.max(np.abs(scores))
            if best > max_score:
                max_score = best
        best_over_draws.append(max_score)
    return float(np.mean(best_over_draws))


if __name__ == "__main__":
    print("=== Rademacher complexity ===\n")
    rng = np.random.default_rng(0)
    n = 200; p = 5
    X = rng.normal(size=(n, p))
    #  Normalise for a clean comparison
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    r_lin_1 = rademacher_linear_ball(X, B=1.0, n_draws=100)
    r_lin_3 = rademacher_linear_ball(X, B=3.0, n_draws=100)
    r_stump = rademacher_stumps(X, n_draws=50)

    print(f"  n={n}, p={p}, rows unit-normalised.\n")
    print(f"  Linear ball ||w||<=1   R_hat = {r_lin_1:.4f}")
    print(f"  Linear ball ||w||<=3   R_hat = {r_lin_3:.4f}   (scales linearly with B)")
    print(f"  Depth-1 stumps         R_hat = {r_stump:.4f}   (much larger, richer class)")

    #  Massart-style theoretical bound: R_hat(F) <= sqrt(2 log |F| / n)
    #  For linear ball, R_hat ~ B / sqrt(n)
    print(f"\n  Sanity: linear-ball R_hat ~ B / sqrt(n) = {1.0 / np.sqrt(n):.4f}   "
          f"(we got {r_lin_1:.4f})")

    print("\n--- library cross-check (custom in R; from-scratch Python) ---")
