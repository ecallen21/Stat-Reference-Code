"""Class-imbalance handling: SMOTE + class weighting + threshold tuning
(Reference §26.17).

Rare-event classification: minority class ~ 1-5%.  Vanilla accuracy is
useless (99% accuracy by predicting all majority); use precision, recall,
F1, PR-AUC, matched-cost objectives.

Three complementary tactics:

1) Class weighting
    Weight minority-class examples in the loss by (n_maj / n_min).
    Available in most classifiers via `class_weight='balanced'`.

2) Resampling
    Random oversampling: duplicate minority rows.  Risk of overfitting.
    Random undersampling: drop majority rows.  Loses information.
    SMOTE (Chawla et al. 2002): synthesize new minority points as
        x_new = x_i + rand * (x_i_neighbor - x_i)
    where x_i_neighbor is a k-NN minority.
    Better than random duplication for numeric features.

3) Threshold tuning
    Default 0.5 threshold optimizes accuracy; tune the score cutoff on
    a held-out set for F1 / precision-at-recall / expected cost.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def smote(X, y, minority_class=1, k: int = 5, n_synth: int = None,
          seed: int = 0) -> tuple:
    """SMOTE oversampling of the minority class."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    rng = np.random.default_rng(seed)
    min_idx = np.where(y == minority_class)[0]
    maj_idx = np.where(y != minority_class)[0]
    if n_synth is None: n_synth = len(maj_idx) - len(min_idx)
    if n_synth <= 0: return X, y
    X_min = X[min_idx]
    synth = []
    for _ in range(n_synth):
        i = int(rng.integers(0, len(X_min)))
        # k-NN among minority
        d = np.linalg.norm(X_min - X_min[i], axis=1); d[i] = np.inf
        nbrs = np.argsort(d)[:k]
        j = int(rng.choice(nbrs))
        gap = rng.uniform(0, 1)
        synth.append(X_min[i] + gap * (X_min[j] - X_min[i]))
    X_synth = np.array(synth); y_synth = np.full(n_synth, minority_class)
    return np.vstack([X, X_synth]), np.concatenate([y, y_synth])


def threshold_metrics(scores, y_true, threshold: float) -> dict:
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    y = np.asarray(y_true)
    tp = int(((y_pred == 1) & (y == 1)).sum()); fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum()); tn = int(((y_pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def tune_threshold(scores, y_true, metric: str = "f1", n_grid: int = 100) -> dict:
    grid = np.linspace(0.01, 0.99, n_grid)
    scores_out = [threshold_metrics(scores, y_true, t)[metric] for t in grid]
    best = int(np.argmax(scores_out))
    return {"threshold_grid": grid, "metric_grid": scores_out,
            "best_threshold": float(grid[best]),
            "best_metric": float(scores_out[best])}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Very imbalanced: 5% positive
    n = 1000
    y = (rng.uniform(0, 1, n) < 0.05).astype(int)
    X = rng.normal(size=(n, 5)); X[y == 1] += 1.5

    print(f"=== Imbalance: {int(y.sum())} positives out of {n} ({y.mean()*100:.1f}%) ===")

    print("\n=== Baseline logistic (naive) ===")
    from scipy.optimize import minimize
    def fit_logistic(X, y, w=None):
        Xd = np.column_stack([np.ones(len(y)), X])
        if w is None: w = np.ones(len(y))
        def neg_ll(b):
            z = Xd @ b
            return -np.sum(w * (y * z - np.logaddexp(0, z)))
        res = minimize(neg_ll, np.zeros(Xd.shape[1]), method="BFGS")
        return res.x
    beta = fit_logistic(X, y)
    scores = 1 / (1 + np.exp(-(np.column_stack([np.ones(n), X]) @ beta)))
    print(f"  at threshold 0.5: {threshold_metrics(scores, y, 0.5)}")

    print("\n=== Class-weighted logistic ===")
    w = np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.0)
    beta_w = fit_logistic(X, y, w=w)
    scores_w = 1 / (1 + np.exp(-(np.column_stack([np.ones(n), X]) @ beta_w)))
    print(f"  at threshold 0.5: {threshold_metrics(scores_w, y, 0.5)}")

    print("\n=== SMOTE resampled + naive logistic ===")
    X_sm, y_sm = smote(X, y, minority_class=1, k=5)
    print(f"  after SMOTE: n = {len(y_sm)}, positives = {int(y_sm.sum())}")
    beta_sm = fit_logistic(X_sm, y_sm)
    scores_sm = 1 / (1 + np.exp(-(np.column_stack([np.ones(n), X]) @ beta_sm)))
    print(f"  at threshold 0.5: {threshold_metrics(scores_sm, y, 0.5)}")

    print("\n=== Threshold tuning on original logistic scores ===")
    t = tune_threshold(scores, y, metric="f1")
    print(f"  best F1 threshold = {t['best_threshold']:.3f}, best F1 = {t['best_metric']:.3f}")
    print(f"  at best threshold: {threshold_metrics(scores, y, t['best_threshold'])}")
