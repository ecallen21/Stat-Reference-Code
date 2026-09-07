"""VC (Vapnik-Chervonenkis) dimension (Reference Sec 46.11).

Vapnik & Chervonenkis 1971. Combinatorial measure of the capacity of
a binary function class F. Definitions:

    A set S = {x_1, ..., x_d} is SHATTERED by F if every one of the
    2^d labelings can be realised by some f in F.
    VC-dim(F) = largest d such that some set of size d is shattered.

Known dimensions:
    * Half-lines in R           :  1
    * Intervals in R            :  2
    * Half-planes in R^p        :  p + 1  (linear classifiers with intercept)
    * Axis-aligned rectangles   :  4
    * Decision stumps in R^p    :  ~ 2 * log2(p) + 1
    * Neural nets               :  O(W * L) for W weights, L layers

Sauer-Shelah lemma: |F restricted to n points| <= (e * n / d)^d
for classes of VC-dim d, giving PAC bounds of order sqrt(d/n) + ...

We estimate VC-dim empirically by ATTEMPTING to shatter random sets
of increasing size: for a class F and set S of size k, enumerate all
2^k labelings and test whether each is realisable.
"""
from __future__ import annotations    # stdlib

from itertools import product    # label enumeration

import numpy as np    # numerical arrays


def shatterable_linear(X_S):
    """Check whether the size-k set X_S (k x p) can be linearly shattered."""
    k, p = X_S.shape
    X1 = np.c_[X_S, np.ones(k)]      # append intercept
    for labels in product([-1, 1], repeat=k):
        y = np.array(labels)
        #  Solve y_i (w' x_i) > 0 by least squares as a proxy: fit w via LS on y
        w, *_ = np.linalg.lstsq(X1, y, rcond=None)
        preds = np.sign(X1 @ w)
        if not np.all(preds == y):
            #  Fall back to a soft-margin check: does an SVM-like separation exist?
            #  A simple LP-style test: solve linear system
            #    Find w s.t. y_i * (w' x_i) >= 1 for all i.
            #  Equivalent to LP; use lstsq on scaled system with slack.
            #  For a compact demo we accept lstsq's answer -- may miss shatterings
            #  near the boundary, so this is a LOWER bound on VC dim.
            return False
    return True


def empirical_vc_lower_bound(class_shatterer, dim, max_size=8, n_trials=15, seed=0):
    """For each candidate size k, try random sets; return largest k that shatters at least once."""
    rng = np.random.default_rng(seed)
    largest_shatterable = 0
    for k in range(1, max_size + 1):
        found = False
        for _ in range(n_trials):
            X_S = rng.normal(size=(k, dim))
            if class_shatterer(X_S):
                found = True
                break
        if found:
            largest_shatterable = k
        else:
            #  If we cannot shatter at this size in many tries, stop searching further.
            break
    return largest_shatterable


def sauer_shelah_bound(n, d):
    """Growth function upper bound |F|_S| <= (e n / d)^d for classes of VC-dim d."""
    if d == 0:
        return 1
    return (np.e * n / d) ** d


if __name__ == "__main__":
    print("=== VC dimension -- empirical lower bounds via shattering ===\n")
    #  For half-planes in R^p, theoretical VC-dim = p + 1.
    for p in [1, 2, 3, 5]:
        d = empirical_vc_lower_bound(shatterable_linear, dim=p,
                                     max_size=p + 3, n_trials=20)
        print(f"  Half-planes in R^{p}:  empirical shattering >= {d}   (theory p+1 = {p + 1})")

    print()
    print("  Sauer-Shelah growth bound |F|_S <= (e n / d)^d for n=100:")
    for d in [1, 3, 5, 10]:
        print(f"    d = {d}:  bound = {sauer_shelah_bound(100, d):.2e}")

    print("\n  Note: lstsq-based shatter check is a lower bound; true VC-dim uses LP/SVM.")

    print("\n--- library cross-check (no direct package; Python custom) ---")
