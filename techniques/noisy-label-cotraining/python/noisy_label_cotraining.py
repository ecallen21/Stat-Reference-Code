"""Co-teaching for noisy labels (Reference Sec 47.98).

Han, Yao, Yu, Niu, Xu, Hu, Tsang & Sugiyama 2018 'Co-teaching:
Robust training of deep neural networks with extremely noisy
labels', NeurIPS. Two networks train on the SAMEmini-batch:

    1. Each ranks its own losses.
    2. Each PEER keeps only the SMALL-LOSS subset (fraction 1 - R(t)).
    3. Each net updates only on the peer's kept subset.

Small-loss examples are more likely to be correctly labelled; peer
selection breaks the confirmation-bias loop of a single net that
memorises its own errors.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.tree import DecisionTreeClassifier    # noise-sensitive base


def train_baseline(X, y, X_test, y_test, n_iters=20, seed=0):
    """Baseline: train once on noisy labels."""
    return DecisionTreeClassifier(max_depth=None, random_state=seed).fit(X, y).score(X_test, y_test)


def coteaching(X, y_noisy, y_true, X_test, y_test, epochs=50, keep_frac=None, seed=0):
    """Simplified co-teaching demo with logistic-regression base learners.

    At each 'epoch' we refit each net on the peer's small-loss subset.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    net_a = DecisionTreeClassifier(max_depth=None, random_state=seed).fit(X, y_noisy)
    net_b = DecisionTreeClassifier(max_depth=None, random_state=seed + 1).fit(X, y_noisy)

    for t in range(epochs):
        # Peer selection: each net's loss on the training set
        loss_a = -np.log(np.clip(net_a.predict_proba(X)[np.arange(n), y_noisy], 1e-9, 1))
        loss_b = -np.log(np.clip(net_b.predict_proba(X)[np.arange(n), y_noisy], 1e-9, 1))
        keep = 1 - min(0.4 * t / epochs, 0.4)     # schedule R(t)
        m = int(keep * n)
        idx_from_b = np.argsort(loss_b)[:m]      # kept-by-b for net-a
        idx_from_a = np.argsort(loss_a)[:m]
        net_a = DecisionTreeClassifier(max_depth=None, random_state=seed).fit(X[idx_from_b], y_noisy[idx_from_b])
        net_b = DecisionTreeClassifier(max_depth=None, random_state=seed + 1).fit(X[idx_from_a], y_noisy[idx_from_a])
    acc = 0.5 * (net_a.score(X_test, y_test) + net_b.score(X_test, y_test))
    return acc


if __name__ == "__main__":
    print("=== Co-teaching for noisy labels (Han et al 2018) ===\n")
    rng = np.random.default_rng(0)

    n, p = 800, 10
    X = rng.normal(size=(n, p))
    coefs = rng.normal(size=p)
    y_true = (X @ coefs > 0).astype(int)
    X_test = rng.normal(size=(400, p))
    y_test = (X_test @ coefs > 0).astype(int)

    for noise in [0.0, 0.1, 0.2, 0.4]:
        y_noisy = y_true.copy()
        flip = rng.random(n) < noise
        y_noisy[flip] = 1 - y_noisy[flip]
        acc_b = train_baseline(X, y_noisy, X_test, y_test)
        acc_c = coteaching(X, y_noisy, y_true, X_test, y_test, epochs=30)
        gain = acc_c - acc_b
        print(f"  Noise rate {noise:.2f}:  baseline acc = {acc_b:.3f}   "
              f"co-teach acc = {acc_c:.3f}   gain = {gain:+.3f}")

    print("\n--- library cross-check (custom in R; cleanlab / noise-detection Python) ---")
