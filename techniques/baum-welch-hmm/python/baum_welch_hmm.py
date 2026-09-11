"""Baum-Welch Algorithm (Reference Sec 47.156).

Baum & Petrie 1966; Baum et al 1970. EM algorithm for HMMs.
E-step: forward-backward gives per-state posteriors gamma_t and
transition posteriors xi_t. M-step: re-estimate pi, A, B by
weighted counts:

    pi     <- gamma_1
    A_ij   <- sum_t xi_t(i,j) / sum_t gamma_t(i)
    B_j(v) <- sum_{t: o_t = v} gamma_t(j) / sum_t gamma_t(j).

Guaranteed non-decreasing likelihood; converges to a local
optimum. Log-space and scaling used to avoid underflow.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def forward_backward(obs, pi, A, B):
    T, K = len(obs), len(pi)
    alpha = np.zeros((T, K)); c = np.zeros(T)                  # scaling factors
    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum(); alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum(); alpha[t] /= c[t]
    beta = np.zeros((T, K)); beta[-1] = 1 / c[-1]
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1]) / c[t]
    log_lik = np.log(c).sum()
    return alpha, beta, log_lik


def baum_welch(obs, K, V, n_iter=50, seed=0, tol=1e-6):
    rng = np.random.default_rng(seed)
    T = len(obs)
    # Random init
    pi = rng.dirichlet(np.ones(K))
    A = np.array([rng.dirichlet(np.ones(K)) for _ in range(K)])
    B = np.array([rng.dirichlet(np.ones(V)) for _ in range(K)])
    lik_hist = []
    for it in range(n_iter):
        alpha, beta, ll = forward_backward(obs, pi, A, B)
        lik_hist.append(ll)
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        # xi via A[i,j] * alpha[t,i] * B[j, o_{t+1}] * beta[t+1, j]
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            xi[t] = (alpha[t][:, None] * A * B[:, obs[t + 1]][None, :] * beta[t + 1][None, :])
            xi[t] /= xi[t].sum()
        # M-step
        pi_new = gamma[0]
        A_new = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        B_new = np.zeros_like(B)
        for v in range(V):
            mask = (obs == v)
            B_new[:, v] = gamma[mask].sum(axis=0) / gamma.sum(axis=0)
        pi, A, B = pi_new, A_new, B_new
        if len(lik_hist) > 1 and abs(lik_hist[-1] - lik_hist[-2]) < tol:
            break
    return {"pi": pi, "A": A, "B": B, "log_lik_history": lik_hist}


if __name__ == "__main__":
    print("=== Baum-Welch algorithm (Baum-Petrie 1966; Baum et al 1970) ===\n")

    # Simulate from a known 2-state / 6-symbol HMM
    pi_t = np.array([0.6, 0.4])
    A_t = np.array([[0.9, 0.1], [0.2, 0.8]])
    B_t = np.array([[1/6] * 6, [0.10, 0.10, 0.10, 0.10, 0.10, 0.50]])
    rng = np.random.default_rng(0)
    T = 800
    z = np.zeros(T, dtype=int); obs = np.zeros(T, dtype=int)
    z[0] = int(rng.choice(2, p=pi_t))
    obs[0] = int(rng.choice(6, p=B_t[z[0]]))
    for t in range(1, T):
        z[t] = int(rng.choice(2, p=A_t[z[t - 1]]))
        obs[t] = int(rng.choice(6, p=B_t[z[t]]))

    r = baum_welch(obs, K=2, V=6, n_iter=100, seed=0)
    ll = r["log_lik_history"]
    print(f"  T = {T} observations, K = 2 hidden states, V = 6 symbols")
    print(f"  Baum-Welch converged in {len(ll)} EM iters (LL: {ll[0]:.1f} -> {ll[-1]:.1f})")
    # Emissions may be permuted; align by matching row 5 (P(6) probability)
    B_est = r["B"]
    perm = [int(np.argmin(B_est[:, 5])), int(np.argmax(B_est[:, 5]))]
    B_aligned = B_est[perm]
    A_aligned = r["A"][np.ix_(perm, perm)]
    print(f"  Truth  A       = {A_t.flatten().round(3)}")
    print(f"  Estim  A (aligned) = {A_aligned.flatten().round(3)}")
    print(f"  Truth  P(6|loaded)  = {B_t[1, 5]:.3f}    Estim = {B_aligned[1, 5]:.3f}")
    print(f"  Truth  P(6|fair)    = {B_t[0, 5]:.3f}    Estim = {B_aligned[0, 5]:.3f}")

    print("\n--- library cross-check (hmmlearn Python; HMM / depmixS4 R) ---")
