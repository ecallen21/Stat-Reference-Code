"""Shapelet Transform for Time-Series Classification (Ref Sec 47.292).

Ye & Keogh 2009 KDD; Hills et al 2014 DAMI. For each candidate
subsequence s of length L, compute its DISTANCE to every training
series (best-matching subseries), then rank candidates by how well
the distance SPLITS classes:

    quality(s) = information_gain(class | dist_to_s)

Retain the top-K shapelets; transform each series to a K-dim
feature vector of shapelet distances; train any tabular classifier.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def znorm(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1)


def dist_to_series(s, T):
    """Min z-normalised Euclidean distance from shapelet s to any position in T."""
    L = len(s); s_n = znorm(s)
    best = np.inf
    for i in range(len(T) - L + 1):
        sub = znorm(T[i:i + L])
        d = np.linalg.norm(sub - s_n)
        if d < best: best = d
    return float(best)


def info_gain(dists, y):
    """Info gain from splitting on the best threshold in dists."""
    y = np.asarray(y)
    p_pos = y.mean()
    ent = -(p_pos * np.log2(p_pos + 1e-9) + (1 - p_pos) * np.log2(1 - p_pos + 1e-9))
    best_ig = 0.0; sorted_d = np.sort(dists)
    for i in range(1, len(sorted_d)):
        t = (sorted_d[i - 1] + sorted_d[i]) / 2
        left = y[dists <= t]; right = y[dists > t]
        if len(left) == 0 or len(right) == 0: continue
        wL = len(left) / len(y); wR = 1 - wL
        entL = -sum(p * np.log2(p + 1e-9) + (1 - p) * np.log2(1 - p + 1e-9)
                     for p in [left.mean()])
        entR = -sum(p * np.log2(p + 1e-9) + (1 - p) * np.log2(1 - p + 1e-9)
                     for p in [right.mean()])
        ig = ent - wL * entL - wR * entR
        if ig > best_ig: best_ig = ig
    return float(best_ig)


def shapelet_transform(X, y, L_choices=(10, 15, 20), n_candidates=30, K=5, rng=None):
    """Return (K shapelets, K-dim feature matrix)."""
    if rng is None: rng = np.random.default_rng(0)
    candidates = []
    for _ in range(n_candidates):
        i = rng.integers(0, len(X))
        L = int(rng.choice(L_choices))
        start = rng.integers(0, len(X[i]) - L)
        candidates.append(X[i][start:start + L])
    # Score each candidate
    quality = []
    for c in candidates:
        d = np.array([dist_to_series(c, T) for T in X])
        quality.append(info_gain(d, y))
    top = np.argsort(quality)[-K:][::-1]
    top_shapelets = [candidates[i] for i in top]
    F = np.array([[dist_to_series(s, T) for s in top_shapelets] for T in X])
    return top_shapelets, F, [quality[i] for i in top]


if __name__ == "__main__":
    print("=== Shapelet Transform (Ye-Keogh 2009; Hills et al 2014) ===\n")
    rng = np.random.default_rng(0)

    # Toy 2-class dataset (short for speed)
    def make(n, freq, noise):
        t = np.linspace(0, 4 * np.pi, 60)
        return [np.sin(freq * t) + noise * rng.standard_normal(60) for _ in range(n)]

    X0 = make(30, 1, 0.3); X1 = make(30, 3, 0.3)
    X = X0 + X1; y = np.array([0] * 30 + [1] * 30)
    from sklearn.model_selection import train_test_split
    Xtr_idx, Xte_idx = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0, stratify=y)
    Xtr = [X[i] for i in Xtr_idx]; Xte = [X[i] for i in Xte_idx]
    ytr = y[Xtr_idx]; yte = y[Xte_idx]

    shapelets, Ftr, quals = shapelet_transform(Xtr, ytr, K=3, n_candidates=20, rng=rng)
    Fte = np.array([[dist_to_series(s, T) for s in shapelets] for T in Xte])
    print(f"  Top-3 shapelets (lengths + info gain):")
    for k in range(3):
        print(f"    shapelet {k+1}: length = {len(shapelets[k])}   info gain = {quals[k]:.3f}")

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression().fit(Ftr, ytr)
    print(f"\n  Train acc:  {clf.score(Ftr, ytr):.3f}")
    print(f"  Test acc:   {clf.score(Fte, yte):.3f}   (2-class 60-obs sine problem)")

    print(f"\n  Shapelets are HUMAN-INSPECTABLE - each is a real subseries")
    print(f"  from the training set that best discriminates classes.")

    print("\n--- library cross-check (tslearn.shapelets; sktime.transformations.panel.shapelets) ---")
