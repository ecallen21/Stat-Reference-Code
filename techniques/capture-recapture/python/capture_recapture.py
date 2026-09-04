"""Capture-recapture (Reference Sec 38.11).

Estimate an unknown population size N from overlapping incomplete
lists / captures.

Classical two-sample:

  LINCOLN-PETERSEN     N_hat = n1 * n2 / m         (biased for small samples)
  CHAPMAN (1951)       N_hat = (n1 + 1)(n2 + 1)/(m + 1) - 1   (bias-corrected)

  Var(N_hat_Chapman)   = (n1+1)(n2+1)(n1-m)(n2-m) / ((m+1)^2 * (m+2))

Multi-sample:

  SCHNABEL             N_hat = sum_t (n_t * M_t) / sum_t m_t
                        where n_t = captured at t, M_t = # marked before t,
                        m_t = recaptured at t.

Assumptions (STRONG!):
  * Closed population (no births/deaths/migration during study).
  * Independent captures, equal capture probability.
  * Marks are not lost between captures.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def lincoln_petersen(n1, n2, m):
    if m == 0:
        return float("inf")
    return n1 * n2 / m


def chapman(n1, n2, m):
    """Chapman bias-corrected estimator + variance."""
    N_hat = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    return {"N_hat": float(N_hat), "SE": float(np.sqrt(var))}


def schnabel(captures):
    """Multi-sample Schnabel estimator.

    captures : list of arrays of caught animal IDs at each occasion.
    """
    marked_before = 0
    seen = set()
    num, den = 0.0, 0.0
    for cap in captures:
        cap = set(int(x) for x in cap)
        n_t = len(cap)
        M_t = marked_before
        m_t = len(cap & seen)
        num += n_t * M_t
        den += m_t
        seen |= cap
        marked_before = len(seen)
    N_hat = num / max(den, 1)
    return {"N_hat": float(N_hat), "recaptures": int(den)}


if __name__ == "__main__":
    print("=== Capture-recapture: Lincoln-Petersen, Chapman, Schnabel ===\n")
    rng = np.random.default_rng(0)
    N_true = 400
    p = 0.25                              # capture prob per occasion
    T = 5                                 # number of occasions
    pop = np.arange(N_true)

    captures = [rng.choice(pop, size=int(N_true * p), replace=False) for _ in range(T)]

    # Two-sample (use first two occasions)
    s1 = set(captures[0]); s2 = set(captures[1])
    n1, n2, m = len(s1), len(s2), len(s1 & s2)
    print(f"  Two occasions: n1 = {n1}, n2 = {n2}, m = {m}")
    lp = lincoln_petersen(n1, n2, m)
    ch = chapman(n1, n2, m)
    print(f"    Lincoln-Petersen  N_hat = {lp:.1f}")
    print(f"    Chapman           N_hat = {ch['N_hat']:.1f}   SE = {ch['SE']:.1f}"
          f"   95%CI = ({ch['N_hat'] - 1.96 * ch['SE']:.1f}, {ch['N_hat'] + 1.96 * ch['SE']:.1f})")

    # Multi-sample Schnabel
    sc = schnabel(captures)
    print(f"\n  Multi-occasion (T = {T}, p = {p}): total recaptures = {sc['recaptures']}")
    print(f"    Schnabel          N_hat = {sc['N_hat']:.1f}   (true = {N_true})\n")

    print("--- library cross-check (R Rcapture/CARE1; Python custom + scipy.optimize) ---")
