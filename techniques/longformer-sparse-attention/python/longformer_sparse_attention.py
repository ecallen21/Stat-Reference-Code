"""Longformer sparse attention (Beltagy, Peters & Cohan 2020).

Restrict attention to a sliding window plus a few GLOBAL
tokens that attend to (and are attended by) everything:

    for each i: attend to i - w/2 ... i + w/2  (local)
    for global tokens g: attend to all, and be attended by all

Complexity O(n * w + n * G) with G global tokens. Enables
sequences up to 4 096 tokens on standard GPUs (16-8× speedup
over quadratic attention).
"""

import numpy as np    # arrays


def full_attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(K.shape[1])
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    return p @ V


def longformer_attention(Q, K, V, window=32, global_idx=()):
    n, d = Q.shape
    out = np.zeros_like(V)
    for i in range(n):
        # local mask
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        idx = list(range(lo, hi)) + list(global_idx)
        idx = sorted(set(idx))
        scores = Q[i] @ K[idx].T / np.sqrt(d)
        p = np.exp(scores - scores.max())
        p = p / p.sum()
        out[i] = p @ V[idx]
    # global tokens attend to everything (already covered above);
    # they also RECEIVE from everything — for query at a global g:
    for g in global_idx:
        scores = Q[g] @ K.T / np.sqrt(d)
        p = np.exp(scores - scores.max())
        p = p / p.sum()
        out[g] = p @ V
    return out


def demo():
    print("=== Longformer sparse attention (Beltagy-Peters-Cohan 2020) ===")
    rng = np.random.default_rng(2026)
    n, d = 512, 32
    Q = rng.standard_normal((n, d)) * 2.0
    K = Q.copy()
    V = rng.standard_normal((n, d))

    out_full = full_attention(Q, K, V)
    for window in [16, 64, 128]:
        for n_global in [0, 4]:
            gidx = list(range(0, n, n // max(1, n_global))) if n_global else ()
            out_lf = longformer_attention(Q, K, V, window=window, global_idx=gidx)
            rel = np.linalg.norm(out_full - out_lf) / np.linalg.norm(out_full)
            pairs = n * (window + len(gidx)) + len(gidx) * n
            print(f"  window={window:3d}, global={len(gidx):2d}: "
                  f"rel err = {rel:.4f}, pairs = {pairs}/{n * n} "
                  f"({100 * pairs / (n * n):.1f}%)")


if __name__ == "__main__":
    demo()
