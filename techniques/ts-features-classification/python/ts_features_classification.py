"""Time series feature extraction and classification (Reference §13.39, §13.41).

Two related tasks:

1) Feature extraction
    Transform each time series into a fixed-length feature vector so
    downstream classifiers / regressors / clustering can treat them as
    ordinary tabular data.  Common features:
        - mean, sd, min, max, median, IQR
        - trend slope (OLS on time)
        - autocorrelation at lags 1, 2, ..., K
        - number of peaks, entropy, complexity (approximate entropy)
        - Fourier / wavelet energy per band
    tsfeatures (R) and tsfresh (Python) are the production toolkits.

2) Time series classification
    Two dominant families:
        - Feature-based: extract features, train a standard classifier
          (random forest / SVM / logistic).
        - Distance-based: 1-NN with DTW distance.  Historically the
          strongest baseline (Bagnall et al. 2017); still hard to beat.

The demo below extracts a small feature set and compares 1-NN DTW vs
1-NN features on a synthetic 3-class problem.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def extract_features(x) -> dict:
    """Fixed-length feature summary of a single time series."""
    x = np.asarray(x, dtype=float); n = len(x)
    d = {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "min": float(x.min()),
        "max": float(x.max()),
        "range": float(x.max() - x.min()),
        "median": float(np.median(x)),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "skewness": float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12)),
        "kurtosis": float(((x - x.mean()) ** 4).mean() / (x.var() ** 2 + 1e-12) - 3),
    }
    # Trend slope
    t = np.arange(n).astype(float)
    d["trend_slope"] = float(np.polyfit(t, x, 1)[0])
    # Autocorrelations
    xc = x - x.mean()
    var = np.dot(xc, xc)
    for lag in (1, 2, 5):
        if n > lag and var > 0:
            d[f"acf_lag{lag}"] = float(np.dot(xc[:-lag], xc[lag:]) / var)
        else:
            d[f"acf_lag{lag}"] = 0.0
    # Number of peaks
    diffs = np.sign(np.diff(x))
    d["n_peaks"] = int(np.sum((diffs[:-1] > 0) & (diffs[1:] < 0)))
    # Spectral energy in low / high band
    Y = np.abs(np.fft.rfft(x - x.mean())) ** 2
    half = len(Y) // 2 or 1
    d["spectral_energy_low"] = float(Y[:half].sum())
    d["spectral_energy_high"] = float(Y[half:].sum())
    return d


def features_matrix(X) -> np.ndarray:
    """Stack a list of series into a (n_series x n_features) matrix."""
    feats = [extract_features(x) for x in X]
    keys = list(feats[0].keys())
    return np.array([[f[k] for k in keys] for f in feats]), keys


def dtw_distance(x, y, window: int = 20) -> float:
    """DTW distance with a Sakoe-Chiba window."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    N, M = len(x), len(y)
    C = np.full((N + 1, M + 1), np.inf); C[0, 0] = 0.0
    for i in range(1, N + 1):
        j_lo = max(1, i - window); j_hi = min(M, i + window)
        for j in range(j_lo, j_hi + 1):
            d = abs(x[i - 1] - y[j - 1])
            C[i, j] = d + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    return float(C[N, M])


def knn_dtw_classify(X_train, y_train, X_test, k: int = 1, window: int = 20) -> np.ndarray:
    """1-NN classifier using DTW distance."""
    preds = []
    for xt in X_test:
        dists = np.array([dtw_distance(xt, xtr, window=window) for xtr in X_train])
        nn = np.argsort(dists)[:k]
        vals, counts = np.unique(y_train[nn], return_counts=True)
        preds.append(vals[np.argmax(counts)])
    return np.array(preds)


def knn_features_classify(F_train, y_train, F_test, k: int = 1) -> np.ndarray:
    """1-NN Euclidean classifier in a fixed feature space (features must be pre-standardized)."""
    preds = []
    for f in F_test:
        d = np.linalg.norm(F_train - f, axis=1)
        nn = np.argsort(d)[:k]
        vals, counts = np.unique(y_train[nn], return_counts=True)
        preds.append(vals[np.argmax(counts)])
    return np.array(preds)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 3 classes: sine, cosine, random walk
    def make_series(cls, T=100):
        t = np.linspace(0, 4 * math.pi, T)
        if cls == 0: return np.sin(t) + rng.normal(0, 0.2, T)
        if cls == 1: return np.cos(t) + rng.normal(0, 0.2, T)
        return np.cumsum(rng.normal(0, 0.3, T))

    n_per = 30
    X = [make_series(c) for c in [0, 1, 2] for _ in range(n_per)]
    y = np.array([c for c in [0, 1, 2] for _ in range(n_per)])
    # Split 80/20
    perm = rng.permutation(len(X))
    tr = perm[:int(0.8 * len(X))]; te = perm[int(0.8 * len(X)):]
    X_tr = [X[i] for i in tr]; y_tr = y[tr]
    X_te = [X[i] for i in te]; y_te = y[te]

    print("=== Feature extraction preview (first series) ===")
    fs = extract_features(X_tr[0])
    for k, v in list(fs.items())[:6]:
        print(f"  {k}: {v:.4f}")

    print("\n=== 1-NN Euclidean on features (standardized) ===")
    F_tr, keys = features_matrix(X_tr)
    F_te, _ = features_matrix(X_te)
    mu = F_tr.mean(0); sd = F_tr.std(0) + 1e-8
    F_tr_z = (F_tr - mu) / sd; F_te_z = (F_te - mu) / sd
    pred = knn_features_classify(F_tr_z, y_tr, F_te_z, k=1)
    print(f"  test accuracy: {(pred == y_te).mean():.3f}")

    print("\n=== 1-NN DTW (window = 15) ===")
    pred = knn_dtw_classify(X_tr, y_tr, X_te, k=1, window=15)
    print(f"  test accuracy: {(pred == y_te).mean():.3f}")

    print("\n--- library cross-check (tsfresh Python; sktime for DTW-1NN) ---")
    try:
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
        clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")
        Xa = np.array(X); Xtr = Xa[tr].reshape(len(tr), 1, -1); Xte = Xa[te].reshape(len(te), 1, -1)
        clf.fit(Xtr, y_tr); acc = (clf.predict(Xte) == y_te).mean()
        print(f"  sktime 1-NN DTW test accuracy: {acc:.3f}")
    except Exception as ex:
        print(f"  (sktime not available: {ex})")
