"""Agreement beyond Cohen's kappa (Reference Sec 38.19).

Cohen's kappa suffers from two paradoxes:
  * KAPPA PARADOX 1 -- very high observed agreement can give low kappa
                        when marginals are extremely unbalanced.
  * KAPPA PARADOX 2 -- symmetric vs asymmetric marginal imbalance
                        gives very different kappa for identical
                        agreement.

Alternatives implemented here:

  PABAK (Byrt et al. 1993)       -- prevalence-adjusted bias-adjusted kappa
                                    = (2 * P_a - 1) for 2 raters, 2 cats
  GWET AC1 (2008)                -- resistant to the kappa paradoxes
  KRIPPENDORFF ALPHA             -- any number of raters, any level of
                                    measurement (nominal, ordinal, interval)
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def cohens_kappa(m):
    """Cohen's kappa from a 2x2 (or KxK) confusion matrix."""
    m = np.asarray(m, dtype=float)
    N = m.sum()
    P_a = np.trace(m) / N
    P_e = (m.sum(axis=1) @ m.sum(axis=0)) / N ** 2
    return (P_a - P_e) / (1 - P_e), P_a, P_e


def pabak(m):
    """Prevalence-adjusted bias-adjusted kappa (2 raters, K cats)."""
    m = np.asarray(m, dtype=float)
    N = m.sum(); K = m.shape[0]
    P_a = np.trace(m) / N
    return (K * P_a - 1) / (K - 1)


def gwet_ac1(m):
    """Gwet's AC1 -- resistant to kappa paradoxes."""
    m = np.asarray(m, dtype=float)
    N = m.sum(); K = m.shape[0]
    P_a = np.trace(m) / N
    pi = (m.sum(axis=1) + m.sum(axis=0)) / (2 * N)     # avg marginal
    P_e = (pi * (1 - pi)).sum() / (K - 1)
    return (P_a - P_e) / (1 - P_e)


def krippendorff_alpha_nominal(ratings):
    """Krippendorff's alpha for nominal data, any number of raters (Krippendorff 2011).

    ratings : (n_items, n_raters) array; use nan for missing.
    """
    R = np.asarray(ratings, dtype=float)
    n_items, _ = R.shape
    cats = np.unique(R[~np.isnan(R)]).astype(int)
    K = len(cats)
    idx = {int(c): i for i, c in enumerate(cats)}
    # Coincidence matrix o[v, c] = sum_i n_iv * (n_ic - [v==c]) / (m_i - 1)
    O = np.zeros((K, K))
    for i in range(n_items):
        row = R[i][~np.isnan(R[i])].astype(int)
        m = len(row)
        if m < 2:
            continue
        counts = np.zeros(K)
        for a in row:
            counts[idx[int(a)]] += 1
        for v in range(K):
            for c in range(K):
                delta = 1 if v == c else 0
                O[v, c] += counts[v] * (counts[c] - delta) / (m - 1)
    n_v = O.sum(axis=1)
    n = n_v.sum()
    # Nominal disagreement: 0 on diagonal, 1 off
    disagreement = 1 - np.eye(K)
    D_o = (O * disagreement).sum() / max(n, 1)
    D_e = ((n_v[:, None] * n_v[None, :]) * disagreement).sum() / max(n * (n - 1), 1)
    return float(1 - D_o / D_e) if D_e > 0 else float("nan")


if __name__ == "__main__":
    print("=== Agreement metrics: kappa, PABAK, Gwet AC1, Krippendorff alpha ===\n")
    # Kappa paradox demonstration
    print("  Kappa paradox 1 (high agreement, low prevalence):")
    m = np.array([[95, 2], [3, 0]])
    k, Pa, Pe = cohens_kappa(m); print(f"    Confusion {m.tolist()}   Cohen k = {k:.3f}   P_a = {Pa:.3f}")
    print(f"    PABAK    = {pabak(m):.3f}")
    print(f"    Gwet AC1 = {gwet_ac1(m):.3f}")

    print("\n  Balanced case with same agreement:")
    m = np.array([[45, 5], [5, 45]])
    k, Pa, Pe = cohens_kappa(m); print(f"    Confusion {m.tolist()}   Cohen k = {k:.3f}   P_a = {Pa:.3f}")
    print(f"    PABAK    = {pabak(m):.3f}")
    print(f"    Gwet AC1 = {gwet_ac1(m):.3f}")

    print("\n  Multi-rater Krippendorff alpha (nominal, 5 raters, 80 items):")
    rng = np.random.default_rng(0)
    # Ground-truth categories + noisy raters
    n_items, n_raters, p_correct = 80, 5, 0.90
    truth = rng.integers(0, 3, size=n_items)
    R = np.stack([np.where(rng.random(n_items) < p_correct, truth, rng.integers(0, 3, size=n_items))
                  for _ in range(n_raters)], axis=1).astype(float)
    print(f"    ratings shape = {R.shape}   (rater accuracy = {p_correct})")
    alpha = krippendorff_alpha_nominal(R)
    print(f"    Krippendorff alpha (nominal) = {alpha:.3f}\n")

    print("--- library cross-check (R irrCAC::gwet.ac1, irr::kripp.alpha; Python krippendorff) ---")
