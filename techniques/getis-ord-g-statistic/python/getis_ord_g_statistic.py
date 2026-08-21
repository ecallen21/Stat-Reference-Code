"""Getis-Ord Gi and Gi* hot-spot statistics (Reference §23.x extra).

Local indicators of spatial association based on the sum of neighbouring values:

    Gi(d)  = sum_{j != i} w_ij(d) x_j / sum_{j != i} x_j
    Gi*(d) = sum_j w_ij(d) x_j / sum_j x_j             (includes i)

Standardised versions have a known Gaussian mean and variance under the
null of complete spatial randomness; Getis & Ord (1992, 1995) give:

    E[Gi*] = W_i / n
    Var[Gi*] = W_i (n - W_i) * s^2 / (n^2 (n - 1) * x_bar^2)

where W_i = sum_j w_ij, s^2 = (1/n) sum(x_j - x_bar)^2.  z-scores > 1.96
indicate significant hot spots (high values clustered), z < -1.96
indicate cold spots.

The Gi* form is preferred (includes the focal cell); permutation p-values
are more reliable than the asymptotic Gaussian on small n / heavy tails.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math (sqrt)

import numpy as np    # numerical arrays + linear algebra


def _distance_band_W(coords, d: float, row_std: bool = False):
    coords = np.asarray(coords, dtype=float); n = len(coords)
    D = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
    W = (D <= d).astype(float)
    if row_std:
        rs = W.sum(axis=1, keepdims=True)
        W = np.where(rs > 0, W / rs, 0.0)
    return W


def getis_ord_gi_star(x, W) -> dict:
    x = np.asarray(x, dtype=float); W = np.asarray(W, dtype=float)
    n = len(x); x_bar = x.mean()
    s2 = ((x - x_bar) ** 2).mean()
    numer = W @ x                                        # includes i via W_ii (should be 1)
    denom = x.sum()
    G_star = numer / denom
    W_i = W.sum(axis=1)
    E = W_i / n
    Var = W_i * (n - W_i) * s2 / (n ** 2 * (n - 1) * x_bar ** 2 + 1e-12)
    z = (G_star - E) / np.sqrt(np.maximum(Var, 1e-12))
    return {"G_star": G_star, "z": z,
            "hotspot": z > 1.96, "coldspot": z < -1.96,
            "method": "Getis-Ord Gi* with asymptotic Gaussian z"}


def getis_ord_permutation(x, W, n_perm: int = 999, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(x); W = np.asarray(W, dtype=float)
    obs = getis_ord_gi_star(x, W)["G_star"]
    ge = np.zeros(n, dtype=int); le = np.zeros(n, dtype=int)
    for _ in range(n_perm):
        xp = rng.permutation(x)
        gp = getis_ord_gi_star(xp, W)["G_star"]
        ge += (gp >= obs); le += (gp <= obs)
    p_hot = (ge + 1) / (n_perm + 1)
    p_cold = (le + 1) / (n_perm + 1)
    p = np.minimum(p_hot, p_cold) * 2                    # two-sided
    return {"G_star": obs, "p_hot": p_hot, "p_cold": p_cold, "p_two_sided": p}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 10 x 10 regular grid, embed a hot cluster in the top-right 3x3
    xs, ys = np.meshgrid(np.arange(10), np.arange(10))
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    n = len(coords)
    x = rng.normal(loc=0, scale=1, size=n)
    hot_mask = (coords[:, 0] >= 7) & (coords[:, 1] >= 7)
    x[hot_mask] += 3.0                                    # +3 SD spike

    # Queen-like distance band (up to sqrt(2))
    W = _distance_band_W(coords, d=math.sqrt(2) + 1e-6, row_std=False)
    # Include focal cell for Gi*
    np.fill_diagonal(W, 1.0)

    res = getis_ord_gi_star(x, W)
    perm = getis_ord_permutation(x, W, n_perm=299, seed=1)

    print(f"=== Getis-Ord Gi* on 10x10 grid with planted hot-cluster ===")
    print(f"  # hot spots (asymp z > 1.96)  : {int(res['hotspot'].sum())}   "
          f"true hot cells: {int(hot_mask.sum())}")
    print(f"  # cold spots (asymp z < -1.96): {int(res['coldspot'].sum())}")

    # confusion vs truth
    tp = int((res["hotspot"] & hot_mask).sum())
    fp = int((res["hotspot"] & ~hot_mask).sum())
    print(f"  hot-spot TP = {tp}, FP = {fp}")

    # permutation-based two-sided p-values
    sig_perm = int((perm["p_two_sided"] < 0.05).sum())
    print(f"  # sig at p<.05 (permutation)  : {sig_perm}")

    # top-5 by z score
    print(f"\n  top-5 cells by z score:")
    top = np.argsort(-res["z"])[:5]
    for k in top:
        print(f"    coords = {coords[k].tolist()}  x = {x[k]:+.2f}  "
              f"Gi* z = {res['z'][k]:+.2f}  "
              f"perm p (two-sided) = {perm['p_two_sided'][k]:.3f}")

    print("\n--- library cross-check (R spdep::localG; pysal.esda.G_Local) ---")
