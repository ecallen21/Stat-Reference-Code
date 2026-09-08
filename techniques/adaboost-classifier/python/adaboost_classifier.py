"""AdaBoost classifier (Reference Sec 47.69).

Freund & Schapire 1997 'A decision-theoretic generalization of on-
line learning and an application to boosting', JCSS 55(1). Iterate:

    1. Fit weak learner h_t on data reweighted by D_t.
    2. Compute weighted error  eps_t = sum D_t(i) * 1{h_t(x_i) != y_i}.
    3. Alpha_t = 0.5 * log((1 - eps_t) / eps_t).
    4. Reweight  D_{t+1}(i) proportional to D_t(i) * exp(-alpha_t * y_i * h_t(x_i))
       (labels in {-1, +1}).

Final F(x) = sign( sum_t alpha_t * h_t(x) ). Interpretable as
forward stagewise additive minimisation of exponential loss.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.tree import DecisionTreeClassifier    # weak learner (stump)


def adaboost(X, y, n_estimators=100):
    """AdaBoost.M1 with decision-stump base learners. y in {-1, +1}."""
    n = len(y)
    D = np.ones(n) / n
    learners = []; alphas = []
    for t in range(n_estimators):
        stump = DecisionTreeClassifier(max_depth=1)
        stump.fit(X, y, sample_weight=D)
        pred = stump.predict(X)
        err = float((D * (pred != y)).sum() / D.sum())
        err = max(err, 1e-10)
        if err >= 0.5:
            break
        alpha = 0.5 * np.log((1 - err) / err)
        D = D * np.exp(-alpha * y * pred)
        D /= D.sum()
        learners.append(stump); alphas.append(alpha)
    return {"learners": learners, "alphas": np.array(alphas)}


def adaboost_predict(model, X):
    F = np.zeros(len(X))
    for h, a in zip(model["learners"], model["alphas"]):
        F += a * h.predict(X)
    return np.sign(F)


if __name__ == "__main__":
    print("=== AdaBoost (Freund-Schapire 1997) ===\n")
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.normal(size=(n, 2))
    # Concentric-ring target -- separable with axis-aligned stumps at multiple splits
    r = np.linalg.norm(X, axis=1)
    y = np.where(r < 1.2, 1.0, -1.0)
    flip = rng.random(n) < 0.05
    y[flip] *= -1

    # Baselines
    from sklearn.tree import DecisionTreeClassifier as DT
    stump_acc = float((DT(max_depth=1).fit(X, y).predict(X) == y).mean())
    tree_acc = float((DT(max_depth=3).fit(X, y).predict(X) == y).mean())

    print(f"  Single stump   train acc = {stump_acc:.3f}")
    print(f"  Depth-3 tree   train acc = {tree_acc:.3f}")

    for T in [10, 50, 200]:
        m = adaboost(X, y, n_estimators=T)
        pred = adaboost_predict(m, X)
        acc = float((pred == y).mean())
        print(f"  AdaBoost T={T:3d}   train acc = {acc:.3f}   ("
              f"{len(m['learners'])} weak learners fit)")

    print("\n--- library cross-check (adabag R; sklearn.ensemble.AdaBoostClassifier Python) ---")
