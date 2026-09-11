"""DiCE - Diverse Counterfactual Explanations (Reference Sec 47.254).

Mothilal, Sharma & Tan 2020 'Explaining Machine Learning
Classifiers Through Diverse Counterfactual Explanations',
FAT*. Given a point x with prediction f(x), find a SET of
counterfactuals {c_1, ..., c_k} that flip the prediction while:

    proximity: each c_i close to x
    diversity: c_i's far from each other
    validity:  each c_i flips the prediction

Optimises a combined loss:
    L = validity + lambda_p * proximity - lambda_d * diversity

More useful than a single counterfactual because it reveals
MULTIPLE valid recourse paths.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def dice_counterfactuals(x, model_predict_proba, target_class=1, k=3, n_iter=200,
                          lambda_p=0.5, lambda_d=1.0, feature_ranges=None, rng=None):
    """Generate k diverse counterfactuals for point x via random-search + local move."""
    if rng is None: rng = np.random.default_rng(0)
    d = len(x)
    if feature_ranges is None:
        feature_ranges = np.tile([[-3.0, 3.0]], (d, 1))
    # Initialise k candidates as jittered copies
    C = x + rng.normal(0, 1.0, size=(k, d))
    for step in range(n_iter):
        # Random perturbation
        C_prop = C + rng.normal(0, 0.3, size=(k, d))
        for i in range(k):
            C_prop[i] = np.clip(C_prop[i], feature_ranges[:, 0], feature_ranges[:, 1])
        # Score both C and C_prop; keep whichever is better under combined loss
        for i in range(k):
            def loss(cand):
                p = model_predict_proba(cand.reshape(1, -1))[0, target_class]
                validity  = max(0, 0.5 - p)                       # want p > 0.5
                proximity = np.abs(cand - x).sum()
                others    = np.delete(C, i, axis=0)
                diversity = np.mean([np.abs(cand - o).sum() for o in others]) if len(others) else 0
                return validity + lambda_p * proximity - lambda_d * diversity
            if loss(C_prop[i]) < loss(C[i]):
                C[i] = C_prop[i]
    return C


if __name__ == "__main__":
    print("=== DiCE - Diverse Counterfactuals (Mothilal et al 2020) ===\n")
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                                    random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)

    # Find a NEGATIVE prediction to flip
    p_te = clf.predict_proba(Xte)[:, 1]
    idx = int(np.argmin(np.abs(p_te - 0.2)))
    x0 = Xte[idx]
    print(f"  Original point x0:  P(class=1) = {clf.predict_proba(x0.reshape(1, -1))[0, 1]:.3f}")
    print(f"  Feature values: {x0.round(2).tolist()}\n")

    ranges = np.column_stack([Xtr.min(0), Xtr.max(0)])
    cfs = dice_counterfactuals(x0, clf.predict_proba, target_class=1, k=3,
                                    n_iter=500, feature_ranges=ranges,
                                    rng=np.random.default_rng(1))
    print(f"  Found {len(cfs)} diverse counterfactuals:")
    for i, c in enumerate(cfs):
        p = clf.predict_proba(c.reshape(1, -1))[0, 1]
        delta = c - x0
        top = int(np.argmax(np.abs(delta)))
        print(f"    CF #{i + 1}:  P(class=1) = {p:.3f}   L1 dist = {np.abs(delta).sum():.2f}   "
              f"biggest change: feat[{top}] += {delta[top]:+.2f}")

    # Pairwise diversity
    dists = [np.abs(cfs[i] - cfs[j]).sum() for i in range(len(cfs)) for j in range(i + 1, len(cfs))]
    print(f"\n  Pairwise L1 diversity between CFs (mean): {np.mean(dists):.2f}")

    print("\n  DiCE reveals multiple recourse routes - user can pick the most actionable.")

    print("\n--- library cross-check (dice-ml Python package) ---")
