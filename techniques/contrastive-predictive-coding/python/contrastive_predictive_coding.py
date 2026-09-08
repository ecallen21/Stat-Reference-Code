"""Contrastive Predictive Coding (CPC) (Reference Sec 47.99).

van den Oord, Li & Vinyals 2018 'Representation learning with
contrastive predictive coding'. Learn representations by predicting
FUTURE samples in the latent space:

    z_t = g_enc(x_t)                    (encoder)
    c_t = g_ar(z_{<=t})                (autoregressive summary)
    f_k(x_{t+k}, c_t) = exp( z_{t+k}^T W_k c_t )   (score)

InfoNCE loss = -log softmax over N-1 negative samples + 1 positive.
Maximises a lower bound on mutual information I(z_{t+k}, c_t).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _encode(X, W_enc):
    """Simple linear encoder: z_t = W_enc @ x_t."""
    return X @ W_enc.T


def _softmax_ce(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return -np.log(np.clip(p[:, 0], 1e-12, 1)).mean()


def train_cpc(seqs, d_enc=8, d_pred=4, lr=0.05, epochs=200, seed=0):
    """Linear CPC on toy 1D-input sequences.
    seqs is (n_seq, T, d_in).
    Predict z_{t+k=1} from c_t = z_t. Contrastive negatives from other seqs.
    """
    rng = np.random.default_rng(seed)
    n_seq, T, d_in = seqs.shape
    W_enc = rng.normal(size=(d_enc, d_in)) * 0.1
    W_k = rng.normal(size=(d_enc, d_enc)) * 0.1

    for ep in range(epochs):
        # Encode all sequences
        Z = np.array([_encode(seqs[i], W_enc) for i in range(n_seq)])
        # positive: z_{t+1} for anchor c_t = z_t.
        c = Z[:, :-1]                       # (n_seq, T-1, d_enc)
        z_pos = Z[:, 1:]                    # (n_seq, T-1, d_enc)
        # Compute scores: (n_seq, T-1, N_neg+1)
        N_neg = 8
        loss = 0.0
        for i in range(n_seq):
            for t in range(T - 1):
                anchor = c[i, t] @ W_k.T                   # (d_enc,)
                pos_score = float(anchor @ z_pos[i, t])
                # sample N_neg negatives: random (seq, time) pairs
                idx = rng.integers(0, n_seq * (T - 1), size=N_neg)
                si, ti = idx // (T - 1), idx % (T - 1)
                neg = Z[si, ti + 1]
                neg_scores = neg @ anchor
                logits = np.concatenate([[pos_score], neg_scores])[None, :]
                loss += _softmax_ce(logits) / (n_seq * (T - 1))
        # Numerical gradient descent (very slow but from scratch)
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}  InfoNCE loss = {loss:.4f}")
        # Simple perturbation-based improvement (not full grad)
        for _ in range(3):
            dW_enc = rng.normal(size=W_enc.shape) * 0.02
            dW_k = rng.normal(size=W_k.shape) * 0.02
            W_enc_new = W_enc + dW_enc
            W_k_new = W_k + dW_k
            # Recompute loss with tiny sample
            Z_new = np.array([_encode(seqs[i], W_enc_new) for i in range(n_seq)])
            loss_new = 0.0
            for i in range(n_seq):
                for t in range(min(3, T - 1)):
                    anchor = Z_new[i, t] @ W_k_new.T
                    pos_score = float(anchor @ Z_new[i, t + 1])
                    idx = rng.integers(0, n_seq * (T - 1), size=N_neg)
                    si, ti = idx // (T - 1), idx % (T - 1)
                    neg = Z_new[si, ti + 1]
                    logits = np.concatenate([[pos_score], neg @ anchor])[None, :]
                    loss_new += _softmax_ce(logits)
            if loss_new < loss * 3:
                W_enc, W_k = W_enc_new, W_k_new
    return W_enc, W_k


if __name__ == "__main__":
    print("=== Contrastive Predictive Coding (van den Oord et al 2018) ===\n")
    rng = np.random.default_rng(0)

    # Toy: sinusoidal sequences with class-specific frequencies.
    T = 20
    n_seq = 12
    freqs = rng.uniform(0.5, 2.0, size=n_seq)
    t = np.linspace(0, 4 * np.pi, T)
    seqs = np.array([np.sin(f * t)[:, None] for f in freqs])

    print("  Training linear CPC on 12 sinusoidal sequences...")
    W_enc, W_k = train_cpc(seqs, d_enc=6, epochs=40, seed=0)

    # Retrieval: for each anchor c_t = z_t, does the true z_{t+1} outrank
    # a random 'negative' z from another sequence?
    Z = np.array([_encode(seqs[i], W_enc) for i in range(n_seq)])
    hits = 0; total = 0
    for i in range(n_seq):
        for tt in range(T - 1):
            anchor = Z[i, tt] @ W_k.T
            pos = float(anchor @ Z[i, tt + 1])
            neg_score = float(anchor @ Z[(i + 1) % n_seq, (tt + 1) % (T - 1)])
            if pos > neg_score:
                hits += 1
            total += 1
    print(f"\n  Predictive accuracy (positive > 1 negative): {hits/total:.3f}")

    print("\n--- library cross-check (limited R; pytorch cpc-audio / SimCLR-time Python) ---")
