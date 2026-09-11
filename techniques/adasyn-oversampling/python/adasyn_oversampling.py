"""ADASYN - Adaptive Synthetic Sampling (Reference Sec 47.248).

He, Bai, Garcia & Li 2008 'ADASYN: Adaptive Synthetic Sampling
Approach for Imbalanced Learning', IJCNN. Like SMOTE, but generates
MORE synthetic samples near HARD minority points (those with more
majority-class neighbours):

    for each x_min:
        r_i = fraction of majority neighbours in k-NN
        g_i ~ r_i normalised over minority
    Number of synths near x_min = g_i * (n_maj - n_min)

Focuses learning on the decision boundary. Compared to SMOTE
(uniform), ADASYN adapts to local difficulty.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def adasyn(X_minority, X_majority, k=5, rng=None):
    """Generate synthetic samples with density scaled to majority-neighbour fraction."""
    if rng is None: rng = np.random.default_rng(0)
    from sklearn.neighbors import NearestNeighbors
    n_needed = len(X_majority) - len(X_minority)
    if n_needed <= 0: return np.zeros((0, X_minority.shape[1]))
    # Combine to find k nearest neighbours in the FULL dataset
    X_all = np.vstack([X_minority, X_majority])
    y_all = np.concatenate([np.ones(len(X_minority)), np.zeros(len(X_majority))])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_all)
    _, indices = nn.kneighbors(X_minority)
    # r_i = fraction of majority-class in k neighbours (excl self)
    r = np.array([(y_all[idx[1:]] == 0).sum() / k for idx in indices])
    if r.sum() == 0: r = np.ones(len(r))
    g = r / r.sum()
    synth = []
    # k-NN within minority for interpolation
    nn_min = NearestNeighbors(n_neighbors=min(k + 1, len(X_minority))).fit(X_minority)
    _, idx_min = nn_min.kneighbors(X_minority)
    for i in range(len(X_minority)):
        n_i = int(round(g[i] * n_needed))
        for _ in range(n_i):
            nbrs = [j for j in idx_min[i][1:] if j != i]
            if not nbrs: continue
            j = int(rng.choice(nbrs))
            alpha = rng.uniform()
            synth.append(X_minority[i] + alpha * (X_minority[j] - X_minority[i]))
    return np.array(synth)


if __name__ == "__main__":
    print("=== ADASYN (He et al 2008 IJCNN) ===\n")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score

    X, y = make_classification(n_samples=2000, n_features=20, n_informative=5,
                                    weights=[0.98, 0.02], flip_y=0.05, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    X_min = Xtr[ytr == 1]; X_maj = Xtr[ytr == 0]
    synth = adasyn(X_min, X_maj, k=5, rng=np.random.default_rng(0))
    print(f"  Train: {len(X_maj)} majority + {len(X_min)} minority")
    print(f"  ADASYN generated {len(synth)} synthetic minority samples")

    X_bal = np.vstack([Xtr, synth])
    y_bal = np.concatenate([ytr, np.ones(len(synth), dtype=int)])
    clf_plain = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    clf_ada = LogisticRegression(max_iter=500).fit(X_bal, y_bal)
    for name, clf in [("plain", clf_plain), ("ADASYN", clf_ada)]:
        yp = clf.predict(Xte)
        print(f"  {name:>7} LR:  P = {precision_score(yte, yp):.3f}   "
              f"R = {recall_score(yte, yp):.3f}   F1 = {f1_score(yte, yp):.3f}")

    # Show adaptive density: r-scores across minorities
    print(f"\n  Minority-difficulty scores (r = fraction of majority in k-NN):")
    from sklearn.neighbors import NearestNeighbors
    X_all = np.vstack([X_min, X_maj]); y_all = np.concatenate([np.ones(len(X_min)), np.zeros(len(X_maj))])
    nn = NearestNeighbors(n_neighbors=6).fit(X_all)
    _, idx = nn.kneighbors(X_min)
    r = np.array([(y_all[i[1:]] == 0).sum() / 5 for i in idx])
    print(f"    mean r = {r.mean():.2f}, max r = {r.max():.2f}   (hard points get more synths)")

    print("\n--- library cross-check (imbalanced-learn.over_sampling.ADASYN) ---")
