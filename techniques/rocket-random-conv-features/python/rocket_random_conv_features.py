"""ROCKET - Random Convolutional Kernel Transform (Ref Sec 47.291).

Dempster, Petitjean & Webb 2020 DAMI. Time-series classifier
that convolves each series with a LARGE random pool of kernels
(typically 10 000), then aggregates each kernel's response via
    - max value
    - PPV (proportion of positive values)
Yielding a 20 000-dim feature vector; a linear (ridge / logistic)
classifier fits on top. State-of-the-art accuracy at a fraction of
the cost of shapelet / HIVE-COTE ensembles.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sample_kernel(rng, length_choices=(7, 9, 11), series_len=100):
    L = int(rng.choice(length_choices))
    weights = rng.standard_normal(L); weights -= weights.mean()
    bias = float(rng.uniform(-1, 1))
    # Random dilation on log scale - cap by series length
    max_exp = np.log2(max(1, (series_len - 1) / (L - 1)))
    dilation = int(2 ** rng.uniform(0, max(0, max_exp)))
    padding = 0 if rng.uniform() > 0.5 else (L - 1) * dilation // 2
    return weights, bias, dilation, padding


def apply_kernel(x, kernel):
    w, b, d, p = kernel
    if p > 0: x = np.pad(x, (p, p))
    L = len(w); out = np.zeros(len(x) - (L - 1) * d)
    for i in range(len(out)):
        idx = i + np.arange(L) * d
        if idx[-1] < len(x): out[i] = np.dot(w, x[idx]) + b
    max_val = float(out.max())
    ppv = float((out > 0).mean())
    return max_val, ppv


def rocket_features(X, n_kernels=200, rng=None, kernels=None):
    """Return (N, 2 * n_kernels) feature matrix (max, PPV per kernel).
    Pass `kernels` to reuse the same kernels across train/test (mandatory
    for a train/test pipeline; RE-sampling kernels defeats the point)."""
    if rng is None: rng = np.random.default_rng(0)
    if kernels is None:
        series_len = len(X[0])
        kernels = [sample_kernel(rng, series_len=series_len) for _ in range(n_kernels)]
    feats = np.zeros((len(X), 2 * len(kernels)))
    for j, k in enumerate(kernels):
        for i in range(len(X)):
            m, p = apply_kernel(X[i], k)
            feats[i, 2 * j] = m; feats[i, 2 * j + 1] = p
    return feats, kernels


if __name__ == "__main__":
    print("=== ROCKET (Dempster et al 2020 DAMI) ===\n")
    rng = np.random.default_rng(0)

    # Toy TSC: 2 classes with different periods
    def make(n, period, noise):
        t = np.linspace(0, 4 * np.pi, 100)
        return np.array([np.sin(period * t + 2 * np.pi * rng.uniform()) + noise * rng.standard_normal(100)
                          for _ in range(n)])

    X0 = make(100, period=1, noise=0.3)
    X1 = make(100, period=3, noise=0.3)
    X = np.vstack([X0, X1]); y = np.array([0] * 100 + [1] * 100)

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import RidgeClassifierCV
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    print(f"  Train: {len(Xtr)}   Test: {len(Xte)}   L=100 per series")

    # Fit kernels ONCE and reuse across train + test
    Ftr, kernels = rocket_features(Xtr, n_kernels=200, rng=rng)
    Fte, _       = rocket_features(Xte, kernels=kernels)
    print(f"  Feature matrix shape: {Ftr.shape}   (2 * n_kernels, same kernels train+test)\n")

    clf = RidgeClassifierCV().fit(Ftr, ytr)
    train_acc = (clf.predict(Ftr) == ytr).mean()
    test_acc = (clf.predict(Fte) == yte).mean()
    print(f"  Ridge accuracy on ROCKET features:  train {train_acc:.3f}   test {test_acc:.3f}\n")

    # Baseline: mean amplitude only (weak feature)
    mean_amp_train = np.abs(Xtr).mean(axis=1)[:, None]
    mean_amp_test = np.abs(Xte).mean(axis=1)[:, None]
    from sklearn.linear_model import LogisticRegression
    baseline = LogisticRegression().fit(mean_amp_train, ytr)
    base_acc = baseline.score(mean_amp_test, yte)
    print(f"  Baseline (mean amplitude only) test accuracy: {base_acc:.3f}")

    print(f"\n  ROCKET's random kernels tap into diverse frequency + scale features,")
    print(f"  giving strong TSC accuracy without any tuning of the feature extractor.")

    print("\n--- library cross-check (sktime.transformations.panel.rocket; tslearn.rocket) ---")
