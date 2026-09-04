"""Dummy / effect / contrast coding (Reference Sec 41.6).

Different ways to encode a k-level categorical predictor in a
regression design matrix.  Each pick changes the interpretation of
the intercept and coefficients but NOT the model fit or predictions.

  REFERENCE / DUMMY   -- one level omitted; each coefficient = mean
                         diff vs the reference.
  EFFECT / DEVIATION  -- coded (-1, 1); each coefficient = diff from
                         GRAND mean.
  HELMERT             -- coefficient j compares level j to the MEAN
                         of levels 1..j-1.
  POLYNOMIAL          -- orthogonal polynomial contrasts (linear,
                         quadratic, ...) for ordered levels.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def dummy_coding(x, drop_first=True):
    levels = sorted(set(x))
    if drop_first:
        levels = levels[1:]
    Z = np.zeros((len(x), len(levels)), dtype=int)
    idx = {lv: i for i, lv in enumerate(levels)}
    for i, xi in enumerate(x):
        if xi in idx:
            Z[i, idx[xi]] = 1
    return Z


def effect_coding(x):
    """Deviation coding: last level -> row of -1s."""
    levels = sorted(set(x))
    K = len(levels)
    idx = {lv: i for i, lv in enumerate(levels)}
    Z = np.zeros((len(x), K - 1))
    for i, xi in enumerate(x):
        j = idx[xi]
        if j == K - 1:
            Z[i, :] = -1
        else:
            Z[i, j] = 1
    return Z, levels


def helmert_coding(x):
    """Helmert: contrast level_j vs average of 1..j-1."""
    levels = sorted(set(x))
    K = len(levels)
    C = np.zeros((K, K - 1))
    for j in range(K - 1):
        for i in range(K):
            if i <= j:
                C[i, j] = -1 / (j + 1)
            elif i == j + 1:
                C[i, j] = 1
    Z = np.zeros((len(x), K - 1))
    idx = {lv: i for i, lv in enumerate(levels)}
    for k, xi in enumerate(x):
        Z[k, :] = C[idx[xi], :]
    return Z, levels


if __name__ == "__main__":
    print("=== Dummy / effect / Helmert coding ===\n")
    x = ["A", "B", "B", "C", "A", "C", "A", "C", "B"]
    print(f"  x = {x}\n")

    D = dummy_coding(x, drop_first=True)
    print(f"  Reference (dummy, drop 'A') matrix:\n{D}\n")

    E, lv = effect_coding(x)
    print(f"  Effect coding (last level = 'C') matrix (levels {lv}):\n{E}\n")

    H, lv = helmert_coding(x)
    print(f"  Helmert coding matrix (levels {lv}):\n{H}\n")

    # Same fit different meaning: regress y on each coding, show equivalence via yhat
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, len(x)) + np.array([0, 1, 1, -1, 0, -1, 0, -1, 1])

    def _fit_yhat(Z):
        X = np.column_stack([np.ones(len(y)), Z])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        return X @ b, b

    for name, Z in [("dummy", D), ("effect", E), ("helmert", H)]:
        yhat, b = _fit_yhat(Z)
        print(f"  {name:>7s}   coefs = {np.round(b, 3)}   yhat[0:3] = {np.round(yhat[:3], 3)}")

    print("\n  All codings give the same yhat; only the coefficient meanings differ.\n")
    print("--- library cross-check (R stats::contr.treatment/sum/helmert/poly; Python patsy C()) ---")
