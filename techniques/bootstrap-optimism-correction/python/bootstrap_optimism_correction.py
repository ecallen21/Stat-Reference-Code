"""Bootstrap optimism correction (Reference Sec 39.4, 39.16).

Efron (1983); Harrell (2015 ch 5); Steyerberg (2019 ch 5).  When you
train and test on the same data you get APPARENT performance which
is too optimistic.  Efron-Harrell bootstrap optimism correction:

  For b = 1..B:
    * draw bootstrap sample S_b (n from n with replacement)
    * fit model on S_b -> M_b
    * perf_boot = perf(M_b on S_b)      -- train
    * perf_orig = perf(M_b on original) -- test
    * optimism_b = perf_boot - perf_orig
  Optimism = mean(optimism_b)
  Corrected performance = apparent - optimism

Handles ANY performance metric; here we use AUC + Brier.
"""
from __future__ import annotations    # stdlib

import warnings                                          # suppress deprecation noise

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression    # baseline logistic

warnings.filterwarnings("ignore")


def _fit(X, y):
    return LogisticRegression(C=1e12, solver="lbfgs", max_iter=500).fit(X, y)


def _auc(y_true, p):
    order = np.argsort(-p)
    y_o = y_true[order]
    pos = y_o == 1; neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(np.cumsum(pos)[neg].sum() / (pos.sum() * neg.sum()))


def _brier(y_true, p):
    return float(((p - y_true) ** 2).mean())


def optimism_correction(X, y, metric_fn, B=200, seed=0):
    """Efron-Harrell bootstrap optimism correction for `metric_fn`."""
    rng = np.random.default_rng(seed)
    n = len(y)
    # Apparent performance (train on all, evaluate on all)
    m = _fit(X, y)
    p_all = m.predict_proba(X)[:, 1]
    apparent = metric_fn(y, p_all)

    optimism = 0.0
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        mb = _fit(Xb, yb)
        p_boot = mb.predict_proba(Xb)[:, 1]
        p_orig = mb.predict_proba(X)[:, 1]
        optimism += metric_fn(yb, p_boot) - metric_fn(y, p_orig)
    optimism /= B
    return {"apparent": float(apparent), "optimism": float(optimism),
            "corrected": float(apparent - optimism), "B": B}


if __name__ == "__main__":
    print("=== Bootstrap optimism correction (Efron-Harrell) ===\n")
    rng = np.random.default_rng(0)
    # Overfit-prone: n = 120, p = 12 (EPV ~ 5), moderate signal
    n, p = 120, 12
    beta = np.concatenate([rng.normal(0, 0.7, 3), np.zeros(p - 3)])
    X = rng.normal(0, 1, (n, p))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ beta - 0.5)))).astype(int)
    print(f"  n = {n}, p = {p}, EPV = {y.sum() / p:.1f}\n")

    for name, metric in [("AUC (higher = better)", _auc), ("Brier (lower = better)", _brier)]:
        oc = optimism_correction(X, y, metric, B=300)
        print(f"  {name}")
        print(f"    apparent  = {oc['apparent']:.3f}")
        print(f"    optimism  = {oc['optimism']:+.3f}")
        print(f"    corrected = {oc['corrected']:.3f}\n")

    print("--- library cross-check (R rms::validate B=200; Python custom + sklearn) ---")
