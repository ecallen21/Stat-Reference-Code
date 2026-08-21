"""Local Moran's I / LISA (Reference §23.4; Anselin 1995).

Decompose global Moran's I into per-location contributions:
    I_i = z_i * sum_j W_ij z_j
    where z_i = (x_i - xbar) / sd(x)

Cluster / outlier categorization
    Compare each x_i vs mean and its neighbours' weighted mean:
        HH (high-high) : x_i > mean and neighbours > mean, I_i > 0  -> hot spot
        LL (low-low)   : x_i < mean and neighbours < mean, I_i > 0  -> cold spot
        HL             : x_i > mean but neighbours < mean, I_i < 0  -> outlier
        LH             : x_i < mean but neighbours > mean, I_i < 0  -> outlier

Permutation p-value: shuffle neighbouring values.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def local_moran(x, W, n_perm: int = 199, seed: int = 0) -> dict:
    x = np.asarray(x, dtype=float); W = np.asarray(W, dtype=float)
    n = len(x); z = (x - x.mean()) / x.std(ddof=1)
    Wz = W @ z
    I_local = z * Wz
    # Cluster / outlier type
    mean_x = x.mean()
    types = []
    for i in range(n):
        neigh_mean = float(W[i] @ x)
        if x[i] > mean_x and neigh_mean > mean_x: types.append("HH")
        elif x[i] < mean_x and neigh_mean < mean_x: types.append("LL")
        elif x[i] > mean_x and neigh_mean < mean_x: types.append("HL")
        else: types.append("LH")
    # Permutation p: shuffle other locations' z, recompute I_i for each i
    rng = np.random.default_rng(seed)
    p_val = np.zeros(n)
    for i in range(n):
        obs = z[i] * float(W[i] @ z)
        perms = []
        others_idx = np.arange(n); others_idx = others_idx[others_idx != i]
        for _ in range(n_perm):
            perm = rng.permutation(others_idx)
            z_shuf = z.copy(); z_shuf[others_idx] = z[perm]
            perms.append(z[i] * float(W[i] @ z_shuf))
        perms = np.array(perms)
        p_val[i] = float((1 + np.sum(np.abs(perms) >= abs(obs))) / (1 + n_perm))
    return {"I_local": I_local, "cluster_type": types, "p_perm": p_val,
            "sig_flag_at_alpha_05": (p_val < 0.05),
            "method": "Local Moran's I / LISA (Anselin 1995)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Grid with a hot-spot cluster in bottom-right
    coords = np.array([(i, j) for i in range(8) for j in range(8)], dtype=float)
    x = rng.normal(0, 1, len(coords))
    hot_idx = [(i, j) for i in (5, 6, 7) for j in (5, 6, 7)]
    for i, j in hot_idx:
        x[i * 8 + j] += 3
    # kNN weights, k=4
    D = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
    W = np.zeros_like(D)
    for i in range(len(coords)):
        d = D[i].copy(); d[i] = np.inf
        idx = np.argsort(d)[:4]; W[i, idx] = 1.0
    W = W / W.sum(1, keepdims=True)

    r = local_moran(x, W, n_perm=99)
    print(f"=== Local Moran's I on hot-spot grid ===")
    n_sig = int(r["sig_flag_at_alpha_05"].sum())
    print(f"  {n_sig} significant local I at alpha 0.05")

    # Report hot-spot cell types
    hh = [k for k, t in enumerate(r["cluster_type"]) if t == "HH" and r["sig_flag_at_alpha_05"][k]]
    ll = [k for k, t in enumerate(r["cluster_type"]) if t == "LL" and r["sig_flag_at_alpha_05"][k]]
    print(f"  significant HH (hot-spot) locations: {len(hh)}")
    print(f"  significant LL (cold-spot) locations: {len(ll)}")

    print("\n--- library cross-check (R spdep::localmoran) ---")
