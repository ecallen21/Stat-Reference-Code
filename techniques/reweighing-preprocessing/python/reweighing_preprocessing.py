"""Reweighing pre-processing (Reference Ch 31 Fairness).

Kamiran & Calders (2012) 'Data Preprocessing Techniques for
Classification without Discrimination.'

Assign per-example weights so that (A, Y) becomes STATISTICALLY
INDEPENDENT in the weighted training set. For each cell (A=a, Y=y):

  w_{a, y}  =  ( P(A=a) * P(Y=y) )  /  P(A=a, Y=y)

This is EXACTLY the odds-ratio inverse: over-represented cells get
weight < 1; under-represented cells get weight > 1. Weighted training
then makes the CLASSIFIER less inclined to reproduce the group-label
correlation from the raw data.

Simple, cheap, model-agnostic PRE-PROCESSING mitigation.

Here we implement the exact weights, train a weighted vs unweighted
logistic regression, and compare demographic-parity + accuracy.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def kamiran_calders_weights(y, groups):
    """Return per-example weights making (A, Y) approximately independent."""
    n = len(y)
    w = np.ones(n)
    for a in np.unique(groups):
        for lab in (0, 1):
            m = (groups == a) & (y == lab)
            n_ay = m.sum()
            if n_ay == 0: continue
            p_a = (groups == a).mean()
            p_y = (y == lab).mean()
            p_ay = n_ay / n
            w[m] = (p_a * p_y) / p_ay
    return w


def logistic_wls(X, y, w=None, lr=0.5, epochs=400, l2=1e-3):
    d = X.shape[1]
    beta = np.zeros(d)
    n = X.shape[0]
    if w is None:
        w = np.ones(n)
    w = w * n / w.sum()      # normalise to keep 'effective' n stable
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        g = X.T @ (w * (p - y)) / n + l2 * beta
        beta -= lr * g
    return beta


def selection_rate(y_hat, mask):
    return float(y_hat[mask].mean()) if mask.any() else float("nan")


def dp_ratio(y_hat, groups):
    r = [selection_rate(y_hat, groups == a) for a in np.unique(groups)]
    return min(r) / max(r) if max(r) > 0 else float("nan")


if __name__ == "__main__":
    print("=== Reweighing pre-processing (Kamiran-Calders 2012) ===\n")
    rng = np.random.default_rng(0)
    n_per = 500
    # Group 0 has higher P(Y=1); features correlate with y AND with group.
    y0 = (rng.random(n_per) < 0.60).astype(int)
    y1 = (rng.random(n_per) < 0.25).astype(int)
    y = np.concatenate([y0, y1])
    groups = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)
    # Feature x0 signals y; x1 correlates with group (proxy variable).
    x0 = y + rng.normal(0, 0.6, len(y))
    x1 = groups + rng.normal(0, 0.4, len(y))
    X = np.stack([x0, x1, np.ones_like(y, dtype=float)], axis=1)   # bias

    w = kamiran_calders_weights(y, groups)
    print("  Kamiran-Calders weights by (A, Y) cell:")
    for a in (0, 1):
        for lab in (0, 1):
            m = (groups == a) & (y == lab)
            print(f"    A={a}, Y={lab}   n={int(m.sum()):>4}   weight={w[m][0]:.3f}")

    beta_erm = logistic_wls(X, y)
    beta_kc = logistic_wls(X, y, w=w)

    y_hat_erm = (_sigmoid(X @ beta_erm) > 0.5).astype(int)
    y_hat_kc  = (_sigmoid(X @ beta_kc)  > 0.5).astype(int)
    print("\n  ERM (no reweighing):")
    print(f"    accuracy={(y_hat_erm == y).mean():.3f}   DP ratio={dp_ratio(y_hat_erm, groups):.3f}")
    print("  Reweighing:")
    print(f"    accuracy={(y_hat_kc == y).mean():.3f}   DP ratio={dp_ratio(y_hat_kc, groups):.3f}")

    print("\n  Reweighing narrows the DP gap at a modest accuracy cost.\n")
    print("--- library cross-check (aif360.algorithms.preprocessing.Reweighing;"
          " fairlearn adjacent samplers) ---")
