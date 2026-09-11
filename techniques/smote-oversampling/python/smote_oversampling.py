"""SMOTE - Synthetic Minority Over-sampling (Reference Sec 47.247).

Chawla, Bowyer, Hall & Kegelmeyer 2002 'SMOTE: Synthetic Minority
Over-sampling Technique', JAIR. Generates synthetic minority-class
samples by interpolating between a minority example and one of
its k nearest minority neighbours:

    for x_min in minority:
        pick a nearest minority neighbour x_nn
        alpha ~ Uniform(0, 1)
        new_sample = x_min + alpha * (x_nn - x_min)

Balances rare-class classification without duplicating exact
copies (which cause overfitting).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def smote(X_minority, n_synth, k=5, rng=None):
    """Generate n_synth synthetic minority samples via SMOTE."""
    if rng is None: rng = np.random.default_rng(0)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_minority))).fit(X_minority)
    _, indices = nn.kneighbors(X_minority)
    synth = []
    for _ in range(n_synth):
        i = int(rng.integers(len(X_minority)))
        nbrs = [j for j in indices[i][1:] if j != i]              # exclude self
        j = int(rng.choice(nbrs))
        alpha = rng.uniform()
        synth.append(X_minority[i] + alpha * (X_minority[j] - X_minority[i]))
    return np.array(synth)


if __name__ == "__main__":
    print("=== SMOTE (Chawla et al 2002 JAIR) ===\n")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, precision_score, recall_score

    X, y = make_classification(n_samples=2000, n_features=20, n_informative=5,
                                    weights=[0.98, 0.02], flip_y=0.05, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    n_min = int((ytr == 1).sum())
    n_maj = int((ytr == 0).sum())
    print(f"  Train: {n_maj} majority + {n_min} minority (imbalance {n_maj/n_min:.1f}:1)")

    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    yp_plain = clf.predict(Xte)
    print(f"  Plain LR:  P = {precision_score(yte, yp_plain):.3f}   "
          f"R = {recall_score(yte, yp_plain):.3f}   F1 = {f1_score(yte, yp_plain):.3f}")

    n_needed = n_maj - n_min
    synth = smote(Xtr[ytr == 1], n_needed, k=5, rng=np.random.default_rng(0))
    X_bal = np.vstack([Xtr, synth])
    y_bal = np.concatenate([ytr, np.ones(len(synth), dtype=int)])
    clf_smote = LogisticRegression(max_iter=500).fit(X_bal, y_bal)
    yp_smote = clf_smote.predict(Xte)
    print(f"  After SMOTE oversample to balance: {n_maj} vs {n_maj}")
    print(f"  SMOTE LR:  P = {precision_score(yte, yp_smote):.3f}   "
          f"R = {recall_score(yte, yp_smote):.3f}   F1 = {f1_score(yte, yp_smote):.3f}")
    print(f"\n  SMOTE typically LIFTS recall (catches more minority) at the cost of")
    print(f"  precision. Net F1 change depends on task; combine with cleaning (Tomek,")
    print(f"  ENN) or use SMOTE-Tomek / SMOTE-ENN for better F1.")

    print("\n--- library cross-check (imbalanced-learn.over_sampling.SMOTE) ---")
