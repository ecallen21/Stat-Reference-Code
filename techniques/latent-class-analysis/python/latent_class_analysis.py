"""Latent class analysis (Reference §19.x extra).

Latent categorical variable K classes indexed by k = 0, ..., K-1.
Each subject i belongs to a latent class C_i with prior probability pi_k.
Given C_i = k, the J binary items are conditionally independent:

    P(U_ij = 1 | C_i = k) = p_{jk}

Log-likelihood:

    log L = sum_i log( sum_k pi_k * prod_j p_{jk}^{U_ij} * (1 - p_{jk})^{1 - U_ij} )

Fit by EM:
  * E: gamma_ik = pi_k * prod_j p_{jk}^{U_ij} (1-p_{jk})^{1-U_ij}, normalise over k.
  * M: pi_k = mean_i gamma_ik;  p_{jk} = sum_i gamma_ik U_ij / sum_i gamma_ik.

BIC / AIC used to choose K.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _e_step(U, pi, p):
    """Return (gamma, log_lik).  gamma: (n, K)."""
    log_pi = np.log(pi + 1e-12)
    logP = np.log(p + 1e-12); log1P = np.log(1 - p + 1e-12)
    # log P(U_i | C=k) = sum_j U_ij log p_jk + (1-U_ij) log(1-p_jk)
    log_lik_ik = (U @ logP.T) + ((1 - U) @ log1P.T) + log_pi[None, :]
    m = log_lik_ik.max(axis=1, keepdims=True)
    ll_row = m.squeeze() + np.log(np.exp(log_lik_ik - m).sum(axis=1))
    gamma = np.exp(log_lik_ik - ll_row[:, None])
    return gamma, float(ll_row.sum())


def fit_lca(U, K: int, n_iter: int = 500, tol: float = 1e-8,
            n_starts: int = 5, seed: int = 0) -> dict:
    U = np.asarray(U, dtype=float); n, J = U.shape
    rng = np.random.default_rng(seed)
    best = {"log_lik": -np.inf}
    for restart in range(n_starts):
        pi = np.full(K, 1.0 / K) + rng.normal(scale=0.01, size=K)
        pi = np.clip(pi, 0.01, None); pi /= pi.sum()
        p = np.clip(rng.uniform(0.15, 0.85, (K, J)), 0.02, 0.98)
        prev_ll = -np.inf
        for it in range(n_iter):
            gamma, ll = _e_step(U, pi, p)
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
            # M-step
            Nk = gamma.sum(axis=0)
            pi = Nk / n
            p = ((gamma.T @ U) / Nk[:, None])
            p = np.clip(p, 1e-4, 1 - 1e-4)
        n_params = (K - 1) + K * J
        bic = -2 * ll + n_params * np.log(n)
        aic = -2 * ll + 2 * n_params
        if ll > best["log_lik"]:
            best = {"log_lik": ll, "pi": pi, "p": p, "gamma": gamma,
                     "n_iter": it + 1, "BIC": float(bic), "AIC": float(aic),
                     "n_params": n_params}
    best["method"] = f"LCA EM (K={K}, best of {n_starts} restarts)"
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 800; J = 6
    # true K=3 classes with well-separated response probs
    pi_true = np.array([0.4, 0.35, 0.25])
    p_true = np.array([[0.9, 0.9, 0.9, 0.1, 0.1, 0.1],       # "yes-yes-no" pattern
                       [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],       # "no-no-yes"
                       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])      # random / balanced

    # simulate
    C = rng.choice(3, n, p=pi_true)
    U = np.zeros((n, J), dtype=int)
    for i in range(n):
        for j in range(J):
            U[i, j] = int(rng.uniform() < p_true[C[i], j])

    print("=== LCA fit (n=800, J=6, true K=3) ===")
    print(f"  {'K':>2}  {'log_lik':>10}  {'BIC':>10}  {'AIC':>10}")
    for K in (1, 2, 3, 4, 5):
        r = fit_lca(U, K=K, n_starts=5, seed=1)
        print(f"  {K:>2}  {r['log_lik']:>10.1f}  {r['BIC']:>10.1f}  {r['AIC']:>10.1f}")

    fit = fit_lca(U, K=3, n_starts=10, seed=2)
    # align estimated to true by matching class means
    order = np.argsort(-fit["p"].sum(axis=1))                # largest-mean-first
    # actually: match on max cosine similarity to true rows
    from itertools import permutations
    best_perm = None; best_diff = np.inf
    for perm in permutations(range(3)):
        diff = np.abs(fit["p"][list(perm)] - p_true).sum()
        if diff < best_diff:
            best_diff = diff; best_perm = perm
    p_aligned = fit["p"][list(best_perm)]
    pi_aligned = fit["pi"][list(best_perm)]

    print(f"\n  best K=3 fit: class prevalences pi_hat = "
          f"{np.round(pi_aligned, 3).tolist()}   true = {pi_true.tolist()}")
    print(f"  class-conditional response probs (rows are classes):")
    for k in range(3):
        print(f"    class {k}: hat = {np.round(p_aligned[k], 3).tolist()}  true = {p_true[k].tolist()}")

    print("\n--- library cross-check (R poLCA::poLCA; Python step_mix / lcmm) ---")
