"""Class-Balanced Loss (Reference Sec 47.252).

Cui, Jia, Lin, Song & Belongie 2019 'Class-Balanced Loss Based
on Effective Number of Samples', CVPR. Reweight loss per class
by the EFFECTIVE NUMBER of samples:

    E_c = (1 - beta^n_c) / (1 - beta)
    w_c = 1 / E_c   (then normalise)

where beta in [0, 1) controls the discount for adding one more
sample. As beta -> 1, w_c -> 1 / n_c (inverse-frequency); as
beta -> 0, w_c -> 1 (uniform). Recommended beta = (N-1)/N.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def class_balanced_weights(class_counts, beta=None):
    """Effective-number reweighting factors, normalised to sum = C."""
    n = np.asarray(class_counts, dtype=float)
    if beta is None: beta = (n.sum() - 1) / n.sum()
    eff = (1.0 - beta ** n) / (1.0 - beta)
    w = 1.0 / eff
    return w * len(n) / w.sum()                                   # scale so mean = 1


if __name__ == "__main__":
    print("=== Class-Balanced Loss (Cui et al 2019 CVPR) ===\n")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score

    X, y = make_classification(n_samples=3000, n_features=15, n_informative=5,
                                    weights=[0.95, 0.05], flip_y=0.05, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    counts = np.bincount(ytr)
    print(f"  Class counts (train): maj={counts[0]}, min={counts[1]}   ({counts[1]/counts.sum()*100:.1f}% minority)")

    for beta in [0.9, 0.99, 0.999, 0.9999]:
        w = class_balanced_weights(counts, beta=beta)
        print(f"  beta = {beta:>6}   weights: maj={w[0]:.3f}, min={w[1]:.3f}   ratio min:maj = {w[1]/w[0]:.1f}")

    # Train sklearn LR with beta-adaptive weights vs inverse-freq vs plain
    print()
    for name, cw in [("plain      ", None),
                     ("inv-freq   ", {0: 1/counts[0], 1: 1/counts[1]}),
                     ("class-bal  ", None)]:
        if name.startswith("class-bal"):
            w = class_balanced_weights(counts, beta=0.999)
            cw = {0: w[0], 1: w[1]}
        clf = LogisticRegression(max_iter=500, class_weight=cw).fit(Xtr, ytr)
        yp = clf.predict(Xte)
        print(f"  {name} LR:  P = {precision_score(yte, yp):.3f}   "
              f"R = {recall_score(yte, yp):.3f}   F1 = {f1_score(yte, yp):.3f}")

    print("\n  Class-balanced softens inverse-frequency by acknowledging that")
    print("  duplicate samples give diminishing returns; commonly beats it.")

    print("\n--- library cross-check (tf.keras / pytorch: manual per-batch weighting) ---")
