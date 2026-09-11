"""Viterbi Algorithm (Reference Sec 47.155).

Viterbi 1967 'Error bounds for convolutional codes and an
asymptotically optimum decoding algorithm', IEEE IT. Dynamic
programming for MAP decoding in an HMM / Markov chain:

    delta_t(j) = max_i delta_{t-1}(i) * a_{i,j} * b_j(o_t)
    psi_t(j)   = argmax_i delta_{t-1}(i) * a_{i,j}

then backtrack from t = T to recover the most likely state
sequence. O(T K^2), same as forward-backward.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def viterbi(obs, pi, A, B):
    """Viterbi decoder: obs (T,), pi (K,), A (K,K), B (K, |V|)."""
    T = len(obs)
    K = len(pi)
    # Log-space to avoid underflow
    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)
    log_B = np.log(B + 1e-300)
    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)
    delta[0] = log_pi + log_B[:, obs[0]]
    for t in range(1, T):
        for j in range(K):
            scores = delta[t - 1] + log_A[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = scores[psi[t, j]] + log_B[j, obs[t]]
    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return {"path": path, "log_prob": float(delta[-1].max())}


if __name__ == "__main__":
    print("=== Viterbi algorithm (Viterbi 1967) ===\n")

    # 2-state fair/loaded-die HMM (Durbin et al 1998 example)
    pi = np.array([0.5, 0.5])
    A = np.array([[0.95, 0.05], [0.10, 0.90]])                # fair->fair 0.95; loaded->loaded 0.90
    B_fair = np.ones(6) / 6                                    # uniform
    B_loaded = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.50]) # heavy on 6
    B = np.stack([B_fair, B_loaded])

    rng = np.random.default_rng(0)
    T = 300
    z = np.zeros(T, dtype=int); obs = np.zeros(T, dtype=int)
    z[0] = int(rng.choice(2, p=pi))
    obs[0] = int(rng.choice(6, p=B[z[0]]))
    for t in range(1, T):
        z[t] = int(rng.choice(2, p=A[z[t - 1]]))
        obs[t] = int(rng.choice(6, p=B[z[t]]))

    r = viterbi(obs, pi, A, B)
    acc = float(np.mean(r["path"] == z))
    print(f"  T = {T} rolls, K = 2 hidden states (fair / loaded die)")
    print(f"  Viterbi decoding accuracy = {acc:.3f}")
    print(f"  Fraction of 'loaded' true / decoded = {z.mean():.3f} / {r['path'].mean():.3f}")
    print(f"  log P*(x, z*) = {r['log_prob']:.2f}")

    # Convex CRF sanity: alternate short-form Viterbi via numpy vectorised
    def viterbi_vec(obs, pi, A, B):
        T, K = len(obs), len(pi)
        lp, la, lb = np.log(pi + 1e-300), np.log(A + 1e-300), np.log(B + 1e-300)
        d = lp + lb[:, obs[0]]
        bp = np.zeros((T, K), dtype=int)
        for t in range(1, T):
            m = d[:, None] + la
            bp[t] = np.argmax(m, axis=0)
            d = np.max(m, axis=0) + lb[:, obs[t]]
        path = np.zeros(T, dtype=int); path[-1] = int(np.argmax(d))
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        return path

    p_vec = viterbi_vec(obs, pi, A, B)
    print(f"  Vectorised implementation agrees on all {T} steps: "
          f"{np.array_equal(r['path'], p_vec)}")

    print("\n--- library cross-check (hmmlearn.hmm.MultinomialHMM.decode; HMM R) ---")
