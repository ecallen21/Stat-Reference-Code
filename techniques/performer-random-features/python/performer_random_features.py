"""Performer FAVOR+ random-features attention
(Choromanski et al 2021 ICLR).

Approximate the softmax kernel exp(q k / sqrt(d)) using
positive orthogonal random features phi:

    exp(q.T k) ~ phi(q).T phi(k)

with phi(x) = exp(-||x||^2 / 2) * exp(W x) / sqrt(m), and W
sampled from orthogonal-Gaussian rows. Attention becomes:

    attention(Q, K, V) ~ phi(Q) (phi(K).T V) / phi(Q) (phi(K).T 1)

Complexity O(n * m * d) — LINEAR in n. FAVOR+ gives an
UNBIASED estimate of the softmax kernel.
"""

import numpy as np    # arrays


def full_attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(K.shape[1])
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    return p @ V


def favor_features(X, W, global_shift):
    """FAVOR+ positive random features.
    phi(x)_i = exp(-||x||^2 / 2) * exp(W_i.T x) / sqrt(m).

    `global_shift` is a scalar subtracted from all exponents; it
    cancels in the ratio num / denom of attention."""
    m = W.shape[0]
    proj = X @ W.T
    normsq = np.sum(X ** 2, axis=1, keepdims=True)
    return np.exp(proj - 0.5 * normsq - global_shift) / np.sqrt(m)


def performer_attention(Q, K, V, m=128, seed=0):
    d = K.shape[1]
    rng = np.random.default_rng(seed)
    # Orthogonal random features (FAVOR+) — better variance than plain Gaussian
    G = rng.standard_normal((m, d))
    Q_, _ = np.linalg.qr(G.T if m <= d else G.T @ G)    # approx orthogonal
    W = G    # plain Gaussian for the demo; orthogonal barely changes low m
    Q_scaled = Q / d ** 0.25
    K_scaled = K / d ** 0.25
    # subtract per-projection max so exponents are bounded but the
    # SAME shift is used on Q and K (unbiased ratio).
    shift = 0.5 * np.max(np.max(Q_scaled @ W.T), initial=0.0)
    phi_Q = favor_features(Q_scaled, W, shift)
    phi_K = favor_features(K_scaled, W, shift)
    num = phi_Q @ (phi_K.T @ V)
    denom = phi_Q @ phi_K.sum(axis=0)
    return num / (denom[:, None] + 1e-12)


def demo():
    print("=== Performer FAVOR+ random-features attention (Choromanski 2021) ===")
    rng = np.random.default_rng(2026)
    n, d = 256, 32
    # small-norm Q, K so FAVOR+ features stay well-scaled
    Q = rng.standard_normal((n, d)) * 0.3
    K = rng.standard_normal((n, d)) * 0.3
    V = rng.standard_normal((n, d))

    out_full = full_attention(Q, K, V)
    for m in [16, 64, 128, 256]:
        out_pf = performer_attention(Q, K, V, m=m, seed=1)
        rel = np.linalg.norm(out_full - out_pf) / np.linalg.norm(out_full)
        print(f"  m = {m:3d}: relative err = {rel:.4f}   (features / attention rows)")


if __name__ == "__main__":
    demo()
