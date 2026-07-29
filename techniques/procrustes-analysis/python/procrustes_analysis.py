"""Procrustes analysis (Reference §9.16).

Given two configurations X and Y of n points in R^p, find the ORTHOGONAL
transformation (rotation + reflection), optional scale, and optional translation
that best aligns Y with X in the least-squares sense:

    minimize   ||X - (s Y Q + t)||_F^2
    over        s (scalar), Q (p x p orthogonal), t (row vector)

Ordinary Procrustes solves the classic centered version (no translation);
GENERALIZED Procrustes aligns >2 configurations simultaneously.

Applications:
    - Shape analysis in morphometrics.
    - Aligning gene-expression or coordinate maps across labs.
    - PCA rotation comparison across bootstrap samples.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def procrustes(X, Y, allow_scale: bool = True) -> dict:
    """Ordinary Procrustes alignment of Y to X.

    Returns rotation Q, scale s, translation t, aligned Y*, and disparity.
    """
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    n, p = X.shape
    mu_x = X.mean(axis=0); mu_y = Y.mean(axis=0)
    Xc = X - mu_x; Yc = Y - mu_y
    # SVD of Yc^T Xc
    M = Yc.T @ Xc
    U, S, Vt = np.linalg.svd(M)
    Q = U @ Vt                                       # optimal rotation/reflection
    if allow_scale:
        s = S.sum() / (Yc ** 2).sum()
    else:
        s = 1.0
    t = mu_x - s * mu_y @ Q
    Y_aligned = s * Y @ Q + t
    disparity = float(((X - Y_aligned) ** 2).sum())
    return {"rotation_Q": Q.tolist(),
            "scale_s": float(s),
            "translation_t": t.tolist(),
            "Y_aligned": Y_aligned.tolist(),
            "disparity_sum_sq_dist": disparity,
            "method": "Ordinary Procrustes alignment"}


def library_versions(X, Y):
    from scipy.spatial import procrustes as sp_procrustes
    mtx1, mtx2, disparity = sp_procrustes(X, Y)
    return {"scipy.spatial.procrustes disparity (normalized)": float(disparity)}


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    n = 30
    X = rng.normal(0, 1, size=(n, 2))
    # Y is X rotated 30 deg + scaled 2x + shifted + noise
    theta = math.radians(30)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    Y = 2.0 * X @ R + np.array([1.5, -0.8]) + rng.normal(0, 0.05, size=(n, 2))

    print("=== Procrustes alignment (should approximately invert 2x rotation) ===")
    r = procrustes(X, Y)
    print(f"  fitted scale s = {r['scale_s']:.4f}   (true = 0.5, i.e. inverse of 2)")
    print(f"  rotation Q =")
    for row in r["rotation_Q"]:
        print(f"    {row}")
    print(f"  disparity (sum of squared distances) = {r['disparity_sum_sq_dist']:.4f}")

    print("\n--- library (scipy) ---")
    for k, v in library_versions(X, Y).items():
        print(f"  {k}: {v}")
