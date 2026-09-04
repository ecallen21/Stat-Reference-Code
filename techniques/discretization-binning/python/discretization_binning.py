"""Discretization / binning (Reference Sec 41.7).

Three families of binning a continuous predictor:

  EQUAL-WIDTH     : split range into k intervals of equal width.
  EQUAL-FREQUENCY : k intervals with roughly equal counts (quantiles).
  ENTROPY-BASED   : split points chosen to maximise information about
                    a target y (Fayyad-Irani 1993 MDL).

Royston-Altman-Sauerbrei 2006 warn that DICHOTOMISING a continuous
predictor throws away information and creates spurious interactions;
they recommend restricted cubic splines instead.  Included here as
demonstration only.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def equal_width(x, k=5):
    edges = np.linspace(x.min(), x.max(), k + 1)
    return np.digitize(x, edges[1:-1])


def equal_freq(x, k=5):
    edges = np.quantile(x, np.linspace(0, 1, k + 1))
    return np.digitize(x, edges[1:-1])


def entropy_binning(x, y, min_gain=0.01):
    """Fayyad-Irani single-cut recursive entropy binning (simplified)."""
    def _entropy(labels):
        _, cnt = np.unique(labels, return_counts=True)
        p = cnt / cnt.sum()
        return -(p * np.log2(p + 1e-12)).sum()

    def _split(idx):
        x_s = x[idx]; y_s = y[idx]
        if len(idx) < 4:
            return []
        cuts = np.unique(x_s)
        best_gain = min_gain
        best_cut = None
        for c in cuts[1:-1]:
            left = y_s[x_s <= c]; right = y_s[x_s > c]
            if len(left) == 0 or len(right) == 0:
                continue
            e_before = _entropy(y_s)
            e_after = (len(left) * _entropy(left) + len(right) * _entropy(right)) / len(y_s)
            gain = e_before - e_after
            if gain > best_gain:
                best_gain = gain; best_cut = c
        if best_cut is None:
            return []
        left_mask = x_s <= best_cut; right_mask = ~left_mask
        return sorted(_split(idx[left_mask]) + [best_cut] + _split(idx[right_mask]))

    cuts = _split(np.arange(len(x)))
    edges = np.array([-np.inf] + list(cuts) + [np.inf])
    bins = np.digitize(x, edges[1:-1])
    return bins, cuts


if __name__ == "__main__":
    print("=== Discretisation: equal-width, equal-frequency, entropy-based ===\n")
    rng = np.random.default_rng(0)
    n = 500
    x = rng.exponential(scale=2, size=n)          # skewed
    y = (x > 2.5).astype(int)                     # true cutoff at 2.5

    ew = equal_width(x, k=5)
    ef = equal_freq(x, k=5)
    eb, cuts = entropy_binning(x, y)

    for name, b in [("equal-width", ew), ("equal-freq", ef), (f"entropy (cuts={[f'{c:.2f}' for c in cuts]})", eb)]:
        print(f"  {name}: bin sizes {np.bincount(b).tolist()}")

    # Discretisation cost: fit logistic on binned vs continuous vs spline features
    from sklearn.linear_model import LogisticRegression
    def _acc(X):
        m = LogisticRegression(C=1e12, solver="lbfgs", max_iter=500).fit(X, y)
        return float(m.score(X, y))
    for name, X in [("continuous", x[:, None]),
                    ("dichot @ median", (x > np.median(x))[:, None].astype(int)),
                    ("dichot @ true 2.5", (x > 2.5)[:, None].astype(int)),
                    ("entropy bins", np.eye(int(eb.max()) + 1)[eb])]:
        print(f"    fit acc [{name:>18s}] = {_acc(X):.3f}")

    print("\n  Royston-Altman-Sauerbrei: dichotomising a continuous variable is a bad idea.\n")
    print("--- library cross-check (R Hmisc::cut2, arules::discretize; Python sklearn.KBinsDiscretizer) ---")
