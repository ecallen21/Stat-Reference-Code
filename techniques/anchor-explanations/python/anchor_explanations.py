"""Anchor explanations (Reference Sec 47.32).

Ribeiro, Singh & Guestrin 2018 'Anchors: high-precision model-agnostic
explanations', AAAI. Extends LIME with an IF-THEN rule ("anchor") whose
precision guarantees the model gives the same prediction WHEN the
anchor's conditions hold, with high probability:

    IF (X_j in interval_j) AND (X_k = category_k) ... THEN pred = c
    with precision >= tau (e.g. 0.95) and largest possible coverage.

Search: beam search over feature-value conditions; precision is
estimated by perturbing non-anchor features and checking model
prediction stability.

Advantage over LIME: crisp rule the user can VERBALISE, and precision
guarantee (unlike LIME's linear approximation).

We implement a simple greedy anchor finder for a tabular classifier.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def perturb_hold(x_star, feature_bins, hold_set, n_samples, rng):
    """Generate n samples matching x_star on `hold_set`; other features drawn iid."""
    d = len(x_star)
    Z = np.zeros((n_samples, d))
    for j in range(d):
        if j in hold_set:
            Z[:, j] = x_star[j]
        else:
            #  Uniform from feature bins
            Z[:, j] = rng.choice(feature_bins[j], size=n_samples)
    return Z


def precision(f, x_star, hold_set, feature_bins, n=200, rng=None):
    """Precision = P(f(z) == f(x_star) | z matches x_star on hold_set)."""
    rng = rng or np.random.default_rng()
    Z = perturb_hold(x_star, feature_bins, hold_set, n, rng)
    y_star = f(x_star)
    return float(np.mean([f(z) == y_star for z in Z]))


def greedy_anchor(f, x_star, feature_bins, tau=0.95, n=200, seed=0):
    """Greedy addition of features to an anchor until precision >= tau."""
    rng = np.random.default_rng(seed)
    d = len(x_star)
    hold_set = set()
    while True:
        best_gain = -1.0; best_j = -1
        for j in range(d):
            if j in hold_set: continue
            trial = hold_set | {j}
            p = precision(f, x_star, trial, feature_bins, n=n, rng=rng)
            if p > best_gain:
                best_gain = p; best_j = j
        hold_set.add(best_j)
        if best_gain >= tau or len(hold_set) == d:
            break
    return {"anchor_features": sorted(hold_set), "precision": best_gain}


if __name__ == "__main__":
    print("=== Anchor explanations (Ribeiro 2018) ===\n")
    #  Toy classifier: y = 1 if x0 > 0.5 AND x2 == 'A' else 0 (x1, x3 irrelevant)
    def f(x):
        return int((x[0] > 0.5) and (x[2] == 0))    # x2==0 is category 'A'

    #  Feature domains
    feature_bins = [
        np.linspace(0, 1, 20),         # x0 continuous
        np.linspace(0, 1, 20),         # x1 irrelevant
        np.array([0, 1, 2]),            # x2 category A/B/C
        np.linspace(0, 1, 20),         # x3 irrelevant
    ]

    x_star = np.array([0.8, 0.5, 0, 0.3])
    print(f"  x* = {x_star.tolist()}   f(x*) = {f(x_star)}")

    r = greedy_anchor(f, x_star, feature_bins, tau=0.95, n=200, seed=0)
    print(f"\n  Greedy anchor:")
    print(f"    Hold features = {r['anchor_features']}")
    print(f"    Precision     = {r['precision']:.3f}")
    print(f"    Verbal rule   : IF x[0]={x_star[0]} AND x[2]={x_star[2]} THEN f = 1  "
          f"(prec {r['precision']:.2f})")

    print("\n--- library cross-check (Marco Ribeiro anchor Python; no direct R port) ---")
