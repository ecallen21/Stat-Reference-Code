"""Tomek Links Undersampling (Reference Sec 47.249).

Tomek 1976 'Two Modifications of CNN', IEEE Trans SMC. A pair
(x_i, x_j) forms a TOMEK LINK if:
    - they are of different classes, and
    - each is the other's nearest neighbour

Removing the majority-class member of each Tomek link CLEANS the
class boundary. Often combined with SMOTE (SMOTE-Tomek) to first
oversample and then clean the boundary.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def tomek_links(X, y):
    """Return indices of majority-class members of Tomek links."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    _, idx = nn.kneighbors(X)
    nearest = idx[:, 1]
    to_remove = []
    for i, j in enumerate(nearest):
        if nearest[j] == i and y[i] != y[j]:
            # majority member is the one whose class has more samples
            n0 = (y == y[i]).sum(); n1 = (y == y[j]).sum()
            to_remove.append(i if n0 > n1 else j)
    return sorted(set(to_remove))


if __name__ == "__main__":
    print("=== Tomek Links Undersampling (Tomek 1976) ===\n")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score

    X, y = make_classification(n_samples=2000, n_features=20, n_informative=5,
                                    weights=[0.90, 0.10], flip_y=0.1, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    n_maj = int((ytr == 0).sum()); n_min = int((ytr == 1).sum())
    print(f"  Train before: {n_maj} majority + {n_min} minority")

    idx_rm = tomek_links(Xtr, ytr)
    keep = np.ones(len(Xtr), dtype=bool); keep[idx_rm] = False
    Xtr_c = Xtr[keep]; ytr_c = ytr[keep]
    print(f"  Removed {len(idx_rm)} Tomek-linked majority samples")
    print(f"  Train after:  {int((ytr_c == 0).sum())} majority + {int((ytr_c == 1).sum())} minority")

    for name, X_use, y_use in [("plain", Xtr, ytr), ("Tomek", Xtr_c, ytr_c)]:
        clf = LogisticRegression(max_iter=500).fit(X_use, y_use)
        yp = clf.predict(Xte)
        print(f"  {name:>6} LR:  P = {precision_score(yte, yp):.3f}   "
              f"R = {recall_score(yte, yp):.3f}   F1 = {f1_score(yte, yp):.3f}")

    print("\n  Tomek links tend to sit ON the decision boundary; removing them")
    print("  produces a cleaner class separation, boosting minority precision.")

    print("\n--- library cross-check (imbalanced-learn.under_sampling.TomekLinks) ---")
