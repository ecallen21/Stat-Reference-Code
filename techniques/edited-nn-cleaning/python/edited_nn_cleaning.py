"""Edited Nearest Neighbours (ENN) Cleaning (Reference Sec 47.250).

Wilson 1972 'Asymptotic Properties of Nearest Neighbor Rules
Using Edited Data', IEEE Trans SMC. Remove any sample x whose
class label disagrees with the MAJORITY vote of its k nearest
neighbours:

    for each x in dataset:
        if class(x) != mode(class(k-NN of x)): remove x

Cleans mislabelled or ambiguous samples. Used in SMOTE-ENN
pipelines: oversample then ENN-clean.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def edited_nn(X, y, k=3):
    """Return boolean mask of samples to KEEP after ENN cleaning."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    keep = np.ones(len(X), dtype=bool)
    for i in range(len(X)):
        neighbours = idx[i][1:]                                   # exclude self
        vote = np.bincount(y[neighbours].astype(int)).argmax()
        if vote != y[i]: keep[i] = False
    return keep


if __name__ == "__main__":
    print("=== Edited Nearest Neighbours (Wilson 1972) ===\n")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score

    X, y = make_classification(n_samples=2000, n_features=20, n_informative=5,
                                    weights=[0.90, 0.10], flip_y=0.15, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    print(f"  Train: {len(Xtr)} samples with 15% label noise (flip_y=0.15)")

    keep = edited_nn(Xtr, ytr, k=3)
    Xtr_c = Xtr[keep]; ytr_c = ytr[keep]
    print(f"  ENN removed {int((~keep).sum())} samples ({100 * (~keep).mean():.1f}%)")
    print(f"  Class breakdown removed: maj={(ytr[~keep] == 0).sum()}, min={(ytr[~keep] == 1).sum()}")

    for name, X_use, y_use in [("plain", Xtr, ytr), ("ENN",   Xtr_c, ytr_c)]:
        clf = LogisticRegression(max_iter=500).fit(X_use, y_use)
        yp = clf.predict(Xte)
        print(f"  {name:>6} LR:  P = {precision_score(yte, yp):.3f}   "
              f"R = {recall_score(yte, yp):.3f}   F1 = {f1_score(yte, yp):.3f}")

    print("\n  ENN cleans noisy labels but can shrink the minority class harshly.")
    print("  Usually paired with SMOTE (SMOTE-ENN: oversample -> ENN clean).")

    print("\n--- library cross-check (imbalanced-learn.under_sampling.EditedNearestNeighbours) ---")
