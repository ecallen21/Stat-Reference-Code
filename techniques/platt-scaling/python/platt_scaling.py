"""Platt Scaling (Reference Sec 47.282).

Platt 1999 MIT Press. Post-hoc calibration for BINARY classifiers
by fitting a logistic regression to the model's SCORES:

    p_calibrated = sigmoid(A * score + B)

Originally developed for SVMs; still standard for binary
classification when only raw scores or logits are available.
Beware of over-fitting on small validation sets - use K-fold
or a held-out calibration split.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def platt_fit(scores, y):
    """Fit sigmoid(A*s + B) via ML with Newton steps."""
    from scipy.optimize import minimize
    def nll(params):
        A, B = params
        p = 1.0 / (1.0 + np.exp(-(A * scores + B)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    res = minimize(nll, x0=[1.0, 0.0], method="BFGS")
    return float(res.x[0]), float(res.x[1])


def platt_predict(scores, A, B):
    return 1.0 / (1.0 + np.exp(-(A * scores + B)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


if __name__ == "__main__":
    print("=== Platt Scaling (Platt 1999) ===\n")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=3000, n_features=10, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
    Xcal, Xte, ycal, yte = train_test_split(Xte, yte, test_size=0.5, random_state=0)

    # Uncalibrated SVM (decision_function, not probabilities)
    svm = LinearSVC().fit(Xtr, ytr)
    s_cal = svm.decision_function(Xcal); s_te = svm.decision_function(Xte)

    # Fit Platt on calibration split
    A, B = platt_fit(s_cal, ycal)
    print(f"  Platt fit: A = {A:.3f}, B = {B:.3f}")

    p_platt = platt_predict(s_te, A, B)
    p_naive = 1.0 / (1.0 + np.exp(-s_te))                        # ad-hoc sigmoid(score)

    # Compare Brier scores
    print(f"\n  Brier on test:")
    print(f"    Naive sigmoid(score):      {brier(p_naive, yte):.4f}")
    print(f"    Platt-scaled probabilities: {brier(p_platt, yte):.4f}")

    # Reliability table (5 bins)
    print(f"\n  Reliability (Platt-scaled):")
    edges = np.quantile(p_platt, np.linspace(0, 1, 6))
    for i in range(5):
        m = (p_platt >= edges[i]) & (p_platt <= edges[i + 1])
        if m.sum() > 0:
            print(f"    bin {i+1}  n = {m.sum():>4}   mean_p = {p_platt[m].mean():.3f}   observed = {yte[m].mean():.3f}")

    # Compare against LR baseline that already outputs probabilities
    lr = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    print(f"\n  Plain LR (already probabilistic) Brier: {brier(p_lr, yte):.4f}")
    print(f"  Platt shines when the base classifier is a margin-based scorer (SVM, gradient-boosted stump).")

    print("\n--- library cross-check (sklearn.calibration.CalibratedClassifierCV(method='sigmoid')) ---")
