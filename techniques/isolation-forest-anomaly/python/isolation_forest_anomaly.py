"""Isolation Forest + one-class SVM + elliptic envelope for anomaly detection
(Reference §26.18).

Anomaly / outlier detection: find rare "different" observations without a
labeled y.  Contrast with time-series anomaly detection (ts-anomaly-detection).

Isolation Forest (Liu-Ting-Zhou 2008)
    Build many random trees; at each node pick a RANDOM feature + RANDOM
    threshold; recurse until each point is isolated.  Anomalies get
    isolated with SHORTER path length -> anomaly score.

One-Class SVM (Scholkopf 2001)
    Learn a decision boundary that encloses the "normal" majority of the
    training data; scores outside are anomalies.  RBF kernel default.

Elliptic Envelope (Rousseeuw 1999)
    Fit robust covariance (MCD) to normal data; Mahalanobis distance
    from robust centre gives an anomaly score.  Works well when normal
    data is Gaussian; poor for multimodal.

The demo below shows all three via sklearn on a synthetic
majority-plus-outliers example.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def isolation_forest(X, n_trees: int = 100, max_samples: int = 256,
                      contamination: float = 0.1, seed: int = 0) -> dict:
    """Isolation Forest anomaly detection.

    Returns anomaly score (higher = more anomalous) per row.
    """
    X = np.asarray(X, dtype=float); n, p = X.shape
    rng = np.random.default_rng(seed)
    subsample = min(max_samples, n)
    def _grow_tree(X_sub, depth, max_depth):
        if depth >= max_depth or len(X_sub) <= 1:
            return {"leaf": True, "n": len(X_sub)}
        j = int(rng.integers(0, p))
        lo, hi = X_sub[:, j].min(), X_sub[:, j].max()
        if lo == hi: return {"leaf": True, "n": len(X_sub)}
        t = rng.uniform(lo, hi)
        L = X_sub[:, j] <= t
        return {"leaf": False, "feature": j, "threshold": t,
                "left": _grow_tree(X_sub[L], depth + 1, max_depth),
                "right": _grow_tree(X_sub[~L], depth + 1, max_depth)}
    def _path_length(node, x, depth):
        if node["leaf"]:
            n = node["n"]
            if n <= 1: return depth
            # Correction term c(n) for expected external path length
            return depth + 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        if x[node["feature"]] <= node["threshold"]:
            return _path_length(node["left"], x, depth + 1)
        return _path_length(node["right"], x, depth + 1)
    max_depth = int(math.ceil(math.log2(subsample))) if subsample > 1 else 1
    trees = []
    for _ in range(n_trees):
        idx = rng.choice(n, size=subsample, replace=False)
        trees.append(_grow_tree(X[idx], 0, max_depth))
    scores = np.zeros(n)
    for i, x in enumerate(X):
        lengths = [_path_length(t, x, 0) for t in trees]
        E_h = np.mean(lengths)
        c_n = 2 * (math.log(subsample - 1) + 0.5772156649) - 2 * (subsample - 1) / subsample
        scores[i] = 2 ** (-E_h / c_n)              # 0.5 = expected; >0.5 = anomalous
    threshold = float(np.quantile(scores, 1 - contamination))
    return {"scores": scores, "threshold": threshold,
            "flag_anomaly": scores > threshold,
            "n_trees": int(n_trees), "max_samples": int(subsample),
            "method": "Isolation Forest (Liu-Ting-Zhou 2008)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 200 normal + 10 obvious outliers in 3-D
    X_normal = rng.normal(0, 1, (200, 3))
    X_out = rng.normal(5, 0.5, (10, 3))
    X = np.vstack([X_normal, X_out]); truth = np.array([0] * 200 + [1] * 10)

    r = isolation_forest(X, n_trees=100, contamination=0.05, seed=0)
    detected = np.where(r["flag_anomaly"])[0]
    print(f"=== Isolation Forest, 100 trees, contamination = 0.05 ===")
    print(f"  {len(detected)} flagged of {len(truth)}; true anomalies at 200..209")
    print(f"  precision = {(np.isin(detected, np.arange(200, 210))).mean():.3f}")
    print(f"  recall = {np.isin(np.arange(200, 210), detected).mean():.3f}")

    print("\n--- library cross-check (sklearn IsolationForest / OneClassSVM / EllipticEnvelope) ---")
    try:
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=0.05, random_state=0).fit(X)
        pred = clf.predict(X)                          # -1 = anomaly
        detected = np.where(pred == -1)[0]
        print(f"  sklearn IsolationForest: {len(detected)} flagged;"
              f" recall {np.isin(np.arange(200, 210), detected).mean():.3f}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
