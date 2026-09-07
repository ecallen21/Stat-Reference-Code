"""Latin hypercube sampling (Reference Sec 45.7).

McKay-Beckman-Conover 1979.  Stratified sampling for
computer-experiment design: divide each of d input dimensions into
n equal-probability bins, then permute across dimensions so that
each bin appears exactly once per dimension.

  * Guarantees uniform 1-D marginals.
  * Empirically better space-filling than plain random sampling.
  * OPTIMAL LATIN HYPERCUBE (Maximin) further maximises min pairwise
    distance.

Used ubiquitously as inputs for expensive simulator emulators
(kriging / Gaussian process surrogates).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def lhs(n, d, seed=0):
    """Standard LHS: n samples in d-dim unit hypercube."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(0, 1, (n, d))
    edges = (np.arange(n)[:, None] + U) / n            # (n, d) stratified positions
    for j in range(d):
        rng.shuffle(edges[:, j])
    return edges


def maximin_lhs(n, d, tries=20, seed=0):
    """Try several LHS realisations, keep the one maximising min-pairwise-distance."""
    best = None
    for s in range(tries):
        X = lhs(n, d, seed=seed + s)
        # min pairwise distance
        diff = X[:, None, :] - X[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        mind = dist.min()
        if best is None or mind > best[0]:
            best = (mind, X)
    return best[1], float(best[0])


if __name__ == "__main__":
    print("=== Latin hypercube sampling (LHS + maximin) ===\n")
    n, d = 10, 3
    X = lhs(n, d)
    print(f"  Standard LHS ({n} pts, {d}-D):")
    print(np.round(X, 3))

    # Marginal uniformity: each bin appears once per column
    bins = (X * n).astype(int)
    print(f"\n  Per-column bin coverage (should be 0..{n - 1}):")
    for j in range(d):
        print(f"    col {j}: {sorted(bins[:, j].tolist())}")

    # Space-filling comparison
    X_lhs, d_lhs = maximin_lhs(30, 5, tries=30)
    rng = np.random.default_rng(0)
    X_rand = rng.uniform(0, 1, (30, 5))
    diff = X_rand[:, None, :] - X_rand[None, :, :]
    d_rand = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(d_rand, np.inf)
    print(f"\n  Min pairwise distance (30 pts in 5-D):")
    print(f"    maximin-LHS = {d_lhs:.3f}")
    print(f"    plain random = {d_rand.min():.3f}\n")

    print("--- library cross-check (R lhs::maximinLHS; Python scipy.stats.qmc.LatinHypercube, pyDOE2) ---")
