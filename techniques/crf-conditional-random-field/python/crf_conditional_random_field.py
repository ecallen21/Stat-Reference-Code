"""Conditional Random Field - CRF (Reference Sec 47.157).

Lafferty, McCallum & Pereira 2001 'Conditional Random Fields:
Probabilistic Models for Segmenting and Labeling Sequence Data',
ICML. Linear-chain CRF models the CONDITIONAL distribution:

    p(y | x) propto exp( sum_t [ f(x_t, y_t) + g(y_{t-1}, y_t) ] ).

Unlike HMMs (which model p(x, y)), CRFs are discriminative and
allow arbitrary overlapping features of x. Train by maximising
conditional log-likelihood; predict via Viterbi. Widely used for
NER, POS tagging, sequence chunking.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def viterbi_crf(emit, trans):
    """Viterbi decoding: emit (T, K) scores, trans (K, K) transitions."""
    T, K = emit.shape
    d = emit[0].copy()
    bp = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        m = d[:, None] + trans + emit[t][None, :]
        bp[t] = np.argmax(d[:, None] + trans, axis=0)
        d = np.max(d[:, None] + trans, axis=0) + emit[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(d))
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def forward_crf(emit, trans):
    """Log-partition function log Z via forward pass."""
    T, K = emit.shape
    log_alpha = emit[0].copy()
    for t in range(1, T):
        # log_alpha_new[j] = logsumexp_i(log_alpha[i] + trans[i, j]) + emit[t, j]
        m = log_alpha[:, None] + trans
        log_alpha = np.log(np.exp(m - m.max(0)).sum(0)) + m.max(0) + emit[t]
    return np.log(np.exp(log_alpha - log_alpha.max()).sum()) + log_alpha.max()


def crf_score(emit, trans, y):
    """Unnormalised path score of tag sequence y."""
    T = emit.shape[0]
    s = emit[0, y[0]]
    for t in range(1, T):
        s += trans[y[t - 1], y[t]] + emit[t, y[t]]
    return s


def crf_train_linear(sequences, tags, K, F, lr=0.1, l2=1e-4, n_epochs=50, seed=0):
    """Linear-chain CRF with F features per token, K tags. Emit(t, k) = W_k^T x_t."""
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.1, size=(K, F))
    T_trans = rng.normal(scale=0.1, size=(K, K))
    for ep in range(n_epochs):
        for X_seq, y_seq in zip(sequences, tags):
            emit = X_seq @ W.T                                  # (T, K)
            # E[feature] under model via forward-backward posteriors
            T, _ = emit.shape
            log_alpha = np.zeros((T, K)); log_alpha[0] = emit[0]
            for t in range(1, T):
                m = log_alpha[t - 1][:, None] + T_trans
                log_alpha[t] = np.log(np.exp(m - m.max(0)).sum(0)) + m.max(0) + emit[t]
            log_Z = np.log(np.exp(log_alpha[-1] - log_alpha[-1].max()).sum()) + log_alpha[-1].max()
            log_beta = np.zeros((T, K))
            for t in range(T - 2, -1, -1):
                m = T_trans + emit[t + 1][None, :] + log_beta[t + 1][None, :]
                log_beta[t] = np.log(np.exp(m - m.max(1, keepdims=True)).sum(1)) + m.max(1)
            gamma = np.exp(log_alpha + log_beta - log_Z)
            # gradient wrt W_k = sum_t (obs_t[y_t = k] - gamma_t(k)) * x_t
            grad_W = np.zeros_like(W)
            for t in range(T):
                obs = np.zeros(K); obs[y_seq[t]] = 1
                grad_W += np.outer(obs - gamma[t], X_seq[t])
            # gradient wrt trans (empirical - expected) — approximate expected as gamma[t-1] outer gamma[t]
            grad_T = np.zeros_like(T_trans)
            for t in range(1, T):
                obs = np.zeros((K, K)); obs[y_seq[t - 1], y_seq[t]] = 1
                grad_T += obs - np.outer(gamma[t - 1], gamma[t])
            W += lr * grad_W - lr * l2 * W
            T_trans += lr * grad_T - lr * l2 * T_trans
    return W, T_trans


if __name__ == "__main__":
    print("=== Conditional Random Field (Lafferty-McCallum-Pereira 2001) ===\n")

    rng = np.random.default_rng(0)
    # Toy chunking task: 3 tags (B, I, O), 4 word-shape features,
    # transitions favour B->I and I->O
    K, F = 3, 4
    A_true = np.array([[0.1, 0.7, 0.2], [0.05, 0.35, 0.6], [0.5, 0.1, 0.4]])
    W_true = np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [-1, -1, 2, 0]])

    def sample_seq(T=20):
        y = np.zeros(T, dtype=int); y[0] = int(rng.choice(3))
        X = np.zeros((T, F))
        for t in range(T):
            if t > 0: y[t] = int(rng.choice(3, p=A_true[y[t - 1]]))
            X[t] = W_true[y[t]] + rng.normal(scale=2.5, size=F)
        return X, y

    train_data = [sample_seq(T=20) for _ in range(200)]
    test_data = [sample_seq(T=20) for _ in range(50)]
    sequences, tags = zip(*train_data)

    W, T_trans = crf_train_linear(list(sequences), list(tags), K=K, F=F,
                                     lr=0.05, n_epochs=15, seed=0)
    # Test decoding
    correct = 0; total = 0; base_correct = 0
    for X, y in test_data:
        emit = X @ W.T
        pred = viterbi_crf(emit, T_trans)
        correct += (pred == y).sum()
        total += len(y)
        # Baseline: per-token argmax on emissions (no transition prior)
        base_correct += (np.argmax(emit, axis=1) == y).sum()
    print(f"  200 training sequences of length 20, K = 3 tags, F = 4 features")
    print(f"  CRF Viterbi token accuracy    = {correct / total:.3f}")
    print(f"  Per-token argmax (no trans)   = {base_correct / total:.3f}")

    print("\n--- library cross-check (sklearn_crfsuite / pystruct Python; CRF R via crfsuite) ---")
