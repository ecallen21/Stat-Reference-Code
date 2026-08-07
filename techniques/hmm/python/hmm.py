"""Hidden Markov Model with categorical emissions (Reference §13.14).

A latent Markov chain S_t in {1, ..., K} emits categorical observations
y_t in {1, ..., M}.

    Pr(S_1 = k)             = pi_k          initial distribution
    Pr(S_t = j | S_{t-1}=i) = A[i, j]       transition matrix
    Pr(y_t = m | S_t = k)   = B[k, m]       emission matrix

Three canonical problems:
    1. FORWARD / BACKWARD   compute p(y_{1:T}) and posteriors.
    2. VITERBI              most-likely single state path S_{1:T}.
    3. BAUM-WELCH           EM estimation of (pi, A, B) from y_{1:T}.

Contrast with Markov-switching model in regime-switching-markov: same math
but continuous Gaussian emissions instead of categorical.  Contrast with
plain Markov chain: here the states are HIDDEN (unobserved).

Applications: speech recognition, POS tagging, gene finding, ion-channel
recordings, activity recognition.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _logsumexp_row(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def forward_backward(y, pi, A, B):
    """Return smoothed state posteriors gamma[t, k] and pairwise xi[t, i, j]."""
    y = np.asarray(y, dtype=int); T = len(y); K = A.shape[0]
    log_pi = np.log(pi + 1e-300); log_A = np.log(A + 1e-300); log_B = np.log(B + 1e-300)
    log_alpha = np.full((T, K), -np.inf); log_beta = np.zeros((T, K))
    log_alpha[0] = log_pi + log_B[:, y[0]]
    for t in range(1, T):
        log_alpha[t] = log_B[:, y[t]] + _logsumexp_row(log_alpha[t - 1][:, None] + log_A, axis=0)
    for t in range(T - 2, -1, -1):
        log_beta[t] = _logsumexp_row(log_A + log_B[:, y[t + 1]][None, :] + log_beta[t + 1][None, :], axis=1)
    log_gamma = log_alpha + log_beta
    log_gamma -= _logsumexp_row(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)
    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        m = log_alpha[t][:, None] + log_A + log_B[:, y[t + 1]][None, :] + log_beta[t + 1][None, :]
        m -= np.max(m)
        m = np.exp(m); xi[t] = m / m.sum()
    ll = float(_logsumexp_row(log_alpha[-1], axis=0))
    return gamma, xi, ll


def viterbi(y, pi, A, B):
    """Return most-likely state path via Viterbi dynamic programming."""
    y = np.asarray(y, dtype=int); T = len(y); K = A.shape[0]
    log_pi = np.log(pi + 1e-300); log_A = np.log(A + 1e-300); log_B = np.log(B + 1e-300)
    delta = np.full((T, K), -np.inf); psi = np.zeros((T, K), dtype=int)
    delta[0] = log_pi + log_B[:, y[0]]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + log_B[:, y[t]]
    path = np.zeros(T, dtype=int); path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = int(psi[t + 1, path[t + 1]])
    return path, float(np.max(delta[-1]))


def baum_welch(y, K: int = 2, M: int = None, max_iter: int = 100, tol: float = 1e-6,
               seed: int = 0) -> dict:
    """EM (Baum-Welch) for a K-state HMM with categorical emissions in {0, ..., M-1}."""
    y = np.asarray(y, dtype=int); T = len(y)
    if M is None: M = int(y.max() + 1)
    rng = np.random.default_rng(seed)
    pi = np.full(K, 1 / K)
    A = rng.dirichlet(np.ones(K) * 5, K)
    B = rng.dirichlet(np.ones(M) * 5, K)
    ll_prev = -np.inf
    for it in range(max_iter):
        gamma, xi, ll = forward_backward(y, pi, A, B)
        # M-step
        pi = gamma[0] / gamma[0].sum()
        A = xi.sum(0) / gamma[:-1].sum(0)[:, None]
        for m in range(M):
            B[:, m] = np.where(y == m, 1, 0) @ gamma
        B /= gamma.sum(0)[:, None]
        if abs(ll - ll_prev) < tol: break
        ll_prev = ll
    return {"initial_dist": pi, "transition_matrix": A, "emission_matrix": B,
            "log_likelihood": float(ll), "iterations": int(it + 1),
            "K_states": int(K), "M_symbols": int(M), "T": int(T),
            "method": "Baum-Welch EM for categorical-emission HMM"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Simulate: 2 states, 3 emission symbols.  State 0 emits mostly 'a', state 1 mostly 'c'.
    K = 2; M = 3
    A_true = np.array([[0.92, 0.08], [0.10, 0.90]])
    B_true = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    pi_true = np.array([0.5, 0.5])
    T = 500
    S = np.zeros(T, dtype=int); S[0] = rng.choice(K, p=pi_true)
    y = np.zeros(T, dtype=int); y[0] = rng.choice(M, p=B_true[S[0]])
    for t in range(1, T):
        S[t] = rng.choice(K, p=A_true[S[t - 1]])
        y[t] = rng.choice(M, p=B_true[S[t]])

    print("=== Baum-Welch fit (K=2, M=3) ===")
    fit = baum_welch(y, K=K, M=M, max_iter=200, seed=1)
    # Align states by emission distribution similarity (label-switching)
    order = [0, 1] if fit["emission_matrix"][0].argmax() == 0 else [1, 0]
    A_est = fit["transition_matrix"][np.ix_(order, order)]
    B_est = fit["emission_matrix"][order]
    print(f"  transition A (est):\n{A_est.round(3)}")
    print(f"  transition A (true):\n{A_true}")
    print(f"  emission B (est):\n{B_est.round(3)}")
    print(f"  emission B (true):\n{B_true}")
    print(f"  log-lik: {fit['log_likelihood']:.3f}, iters: {fit['iterations']}")

    print("\n=== Viterbi decoding accuracy ===")
    path, _ = viterbi(y, fit["initial_dist"], fit["transition_matrix"], fit["emission_matrix"])
    # Re-align path to true state labels
    remap = {order[0]: 0, order[1]: 1}
    path_aligned = np.array([remap[p] for p in path])
    acc = max((path_aligned == S).mean(), (path_aligned == 1 - S).mean())
    print(f"  Viterbi state accuracy: {acc:.3f}")

    print("\n--- library cross-check (hmmlearn) ---")
    try:
        from hmmlearn.hmm import CategoricalHMM
        m = CategoricalHMM(n_components=K, n_iter=200, random_state=0)
        m.fit(y.reshape(-1, 1))
        print(f"  hmmlearn log-lik: {m.score(y.reshape(-1, 1)):.3f}")
    except Exception as ex:
        print(f"  (hmmlearn not available: {ex})")
