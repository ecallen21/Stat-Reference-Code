"""Balanced incomplete block design (BIBD) (Reference Sec 32.9).

Yates 1936; Fisher 1935. A block design where each block sees only
a subset of the v treatments (block size k < v), but every PAIR of
treatments appears together in the same number lambda of blocks.
Parameters (v, b, r, k, lambda) satisfy:

    r * (k - 1) = lambda * (v - 1)         (each treatment: r blocks * k-1 partners = lambda * (v-1))
    b * k       = r * v                     (total plot-count identity)

Analysis: OLS on plot data with block + treatment factors gives
intra-block treatment estimates:

    Q_i = T_i - (1 / k) sum_{blocks containing i} B_j
    tau_hat_i = k * Q_i / (lambda * v)

We construct a Fisher (v = 7, b = 7, r = 3, k = 3, lambda = 1)
BIBD (the Fano plane), simulate outcomes with treatment effects,
and recover the effects.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


#  Fano-plane BIBD (v = 7, b = 7, k = 3, r = 3, lambda = 1)
FANO = np.array([
    [0, 1, 2],
    [0, 3, 4],
    [0, 5, 6],
    [1, 3, 5],
    [1, 4, 6],
    [2, 3, 6],
    [2, 4, 5],
])


def intra_block_estimator(y, blocks_list, v):
    """y : one obs per (block, plot) row -- length b*k.
    blocks_list : list of arrays of treatment indices per block."""
    b = len(blocks_list); k = len(blocks_list[0])
    r = b * k // v
    lam = r * (k - 1) // (v - 1)

    T = np.zeros(v)                # treatment totals
    B = np.zeros(b)                 # block totals
    counts = np.zeros(v, dtype=int)
    idx = 0
    for j in range(b):
        for i in blocks_list[j]:
            T[i] += y[idx]; B[j] += y[idx]; counts[i] += 1
            idx += 1

    #  Adjusted treatment totals Q_i
    Q = T.copy()
    idx = 0
    #  Compute per-block sum: contribution to Q_i is subtract B_j / k for each block containing i
    Q = np.zeros(v)
    for j, treats in enumerate(blocks_list):
        for i in treats:
            Q[i] += - B[j] / k
    Q = T + Q
    #  tau_hat_i
    tau_hat = k * Q / (lam * v)
    return {"T": T, "B": B, "Q": Q, "tau_hat": tau_hat,
            "params": {"v": v, "b": b, "r": r, "k": k, "lambda": lam}}


if __name__ == "__main__":
    print("=== Balanced incomplete block design (Fano-plane BIBD) ===\n")
    rng = np.random.default_rng(0)
    v = 7
    blocks_list = [list(row) for row in FANO]
    #  True treatment effects (centred) + block effects + noise
    tau_true = np.array([2, 1, 0, -1, -2, 1, -1], dtype=float)
    tau_true -= tau_true.mean()
    beta_true = rng.normal(0, 1, size=len(blocks_list))
    y = []
    for j, treats in enumerate(blocks_list):
        for i in treats:
            y.append(10 + tau_true[i] + beta_true[j] + rng.normal(scale=0.3))
    y = np.array(y)

    r = intra_block_estimator(y, blocks_list, v)
    print(f"  BIBD parameters: {r['params']}\n")
    print(f"  True tau  = {tau_true.round(2).tolist()}")
    print(f"  Est tau   = {(r['tau_hat'] - r['tau_hat'].mean()).round(2).tolist()}")

    print("\n  Every pair of treatments (i, j) appears together in exactly lambda = 1 block,")
    print(f"  so pairwise contrasts have EQUAL variance -- the design's key balance property.")

    print("\n--- library cross-check (AlgDesign / crossdes R; pyDOE2 Python) ---")
