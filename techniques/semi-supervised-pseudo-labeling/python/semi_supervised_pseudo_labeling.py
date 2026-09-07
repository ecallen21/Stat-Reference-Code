"""Semi-supervised pseudo-labelling (Reference Sec 47.39).

Lee 2013 'Pseudo-label: the simple and efficient semi-supervised
learning method for deep neural networks'. Iteratively:

    1. Train the classifier on labelled data L.
    2. Predict labels on unlabelled data U.
    3. Add high-CONFIDENCE predictions (max prob > tau) as
       pseudo-labels to L.
    4. Retrain.

Also related: self-training (Yarowsky 1995), co-training (Blum &
Mitchell 1998), FixMatch (Sohn 2020) with strong / weak augmentation.

We implement the basic pseudo-label loop for a logistic classifier
and show accuracy improves as pseudo-labels are added.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fit_logistic(X, y):
    Xd = np.c_[np.ones(len(X)), X]
    beta = np.zeros(Xd.shape[1])
    for _ in range(200):
        p = 1 / (1 + np.exp(-Xd @ beta))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        grad = Xd.T @ (y - p)
        H = Xd.T @ np.diag(p * (1 - p)) @ Xd
        beta += np.linalg.solve(H + 1e-3 * np.eye(Xd.shape[1]), grad)
    return beta


def predict_proba(X, beta):
    return 1 / (1 + np.exp(-np.c_[np.ones(len(X)), X] @ beta))


if __name__ == "__main__":
    print("=== Semi-supervised pseudo-labelling (Lee 2013) ===\n")
    rng = np.random.default_rng(0)
    n_labelled = 30; n_unlabelled = 500; n_test = 300; d = 3
    beta_true = np.array([1.0, -0.5, 0.7])

    X = rng.normal(size=(n_labelled + n_unlabelled + n_test, d))
    y = ((X @ beta_true + rng.normal(scale=0.5, size=len(X))) > 0).astype(int)

    Xl, yl = X[:n_labelled], y[:n_labelled]
    Xu = X[n_labelled: n_labelled + n_unlabelled]
    Xte, yte = X[-n_test:], y[-n_test:]

    #  Baseline: labelled-only
    beta0 = fit_logistic(Xl, yl)
    acc0 = float(np.mean((predict_proba(Xte, beta0) > 0.5) == yte))
    print(f"  Labelled-only baseline accuracy = {acc0:.3f}   (n = {n_labelled})\n")

    #  Pseudo-labelling loop
    Xl_cur, yl_cur = Xl.copy(), yl.copy()
    beta = beta0.copy()
    for it in range(1, 6):
        p_u = predict_proba(Xu, beta)
        conf = np.maximum(p_u, 1 - p_u)
        tau = 0.9
        confident = conf > tau
        Xp = Xu[confident]
        yp = (p_u[confident] > 0.5).astype(int)
        Xl_new = np.r_[Xl_cur, Xp]; yl_new = np.r_[yl_cur, yp]
        beta = fit_logistic(Xl_new, yl_new)
        acc = float(np.mean((predict_proba(Xte, beta) > 0.5) == yte))
        print(f"  Iter {it}: {confident.sum()} pseudo-labels added   "
              f"total = {len(yl_new)}   test acc = {acc:.3f}")

    print("\n  Note: pseudo-labelling can HELP or HURT depending on baseline quality.")
    print("  Here the labelled classifier is already good (0.86) so gains are small;")
    print("  confirmation-bias risk means poorly-calibrated confidences reinforce errors.")

    print("\n--- library cross-check (self-training scikit-learn SelfTrainingClassifier Python) ---")
