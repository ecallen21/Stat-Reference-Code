"""Record linkage / entity resolution (Reference Sec 38.6).

Fellegi-Sunter (1969) probabilistic record linkage.  Given two files
A and B and comparison vectors gamma over shared fields, compute the
LIKELIHOOD RATIO

  R(gamma) = P(gamma | match) / P(gamma | non-match)
           = product over fields of m_i^gamma_i * (1 - m_i)^(1 - gamma_i)
             / u_i^gamma_i * (1 - u_i)^(1 - gamma_i)

  m_i = P(field i agrees | true match).       Usually high (0.9 typ).
  u_i = P(field i agrees | true non-match).   Usually low (rare
                                              agreement by chance).

Choose thresholds T_lo < T_hi so that
  * R > T_hi : declare MATCH
  * R < T_lo : declare NON-MATCH
  * else     : send to clerical review

Parameters (m, u) can be pre-set or EM-estimated (Winkler 1988).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fellegi_sunter_weights(m, u):
    """Log-weights for agree / disagree per field (base-2 by convention)."""
    m = np.asarray(m, dtype=float)
    u = np.asarray(u, dtype=float)
    w_agree = np.log2(m / u)
    w_disagree = np.log2((1 - m) / (1 - u))
    return w_agree, w_disagree


def classify(gamma, m, u, T_lo, T_hi):
    """Classify a comparison vector as match / possible / non-match."""
    w_a, w_d = fellegi_sunter_weights(m, u)
    W = np.where(gamma == 1, w_a, w_d).sum(axis=-1)
    label = np.where(W > T_hi, "match", np.where(W < T_lo, "non-match", "possible"))
    return {"weight": W, "label": label}


def em_mu_estimate(gammas, n_iter=50, seed=0):
    """EM for (m, u, pi) given comparison vectors (Winkler 1988)."""
    rng = np.random.default_rng(seed)
    n, k = gammas.shape
    m = np.full(k, 0.9)
    u = np.full(k, 0.1)
    pi = 0.1                             # match prior
    for _ in range(n_iter):
        # E-step: responsibilities
        lik_m = np.prod(m ** gammas * (1 - m) ** (1 - gammas), axis=1)
        lik_u = np.prod(u ** gammas * (1 - u) ** (1 - gammas), axis=1)
        num = pi * lik_m
        den = num + (1 - pi) * lik_u + 1e-300
        gM = num / den                    # P(match | gamma)
        # M-step
        m = (gM[:, None] * gammas).sum(axis=0) / (gM.sum() + 1e-12)
        u = ((1 - gM)[:, None] * gammas).sum(axis=0) / ((1 - gM).sum() + 1e-12)
        pi = gM.mean()
        m = np.clip(m, 1e-4, 1 - 1e-4)
        u = np.clip(u, 1e-4, 1 - 1e-4)
    return {"m": m, "u": u, "pi": float(pi)}


if __name__ == "__main__":
    print("=== Record linkage: Fellegi-Sunter with EM parameter estimation ===\n")
    rng = np.random.default_rng(0)
    fields = ["first_name", "last_name", "dob", "zip"]

    # Simulate 500 pairs: 100 true matches, 400 non-matches.
    n_m, n_u = 100, 400
    m_true = np.array([0.95, 0.98, 0.99, 0.90])
    u_true = np.array([0.05, 0.02, 0.01, 0.05])

    gammas_m = (rng.random((n_m, 4)) < m_true).astype(int)
    gammas_u = (rng.random((n_u, 4)) < u_true).astype(int)
    gammas = np.vstack([gammas_m, gammas_u])
    truth = np.array([1] * n_m + [0] * n_u)

    est = em_mu_estimate(gammas, n_iter=100)
    print(f"  EM estimates    m = {np.round(est['m'], 3)}   u = {np.round(est['u'], 3)}"
          f"   pi = {est['pi']:.3f}")
    print(f"  Truth          m = {m_true}   u = {u_true}   pi = {n_m / (n_m + n_u):.3f}")

    # Classify with EM-estimated params
    res = classify(gammas, est["m"], est["u"], T_lo=-5, T_hi=5)
    labels = res["label"]
    print(f"\n  Classification summary:")
    for lab in ["match", "possible", "non-match"]:
        mask = labels == lab
        n_true_match = (truth[mask] == 1).sum()
        n_true_non = (truth[mask] == 0).sum()
        print(f"    {lab:>10s}  n = {mask.sum():>3d}   (true matches {n_true_match:>3d}"
              f", true non-matches {n_true_non:>3d})")

    print("\n--- library cross-check (R fastLink/RecordLinkage; Python recordlinkage/dedupe) ---")
