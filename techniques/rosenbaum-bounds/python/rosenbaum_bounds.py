"""Rosenbaum bounds for sensitivity to unmeasured confounding (Reference Sec 15.23).

Rosenbaum 2002.  For matched-pair studies: how strong an
unmeasured confounder U would need to be (measured as an odds
ratio Gamma on treatment assignment within matched pairs) to
overturn the study's conclusion?

For each Gamma >= 1, the McNemar-style test on discordant pairs
uses bounds:
  p_max = Gamma / (1 + Gamma)     upper bound on Pr(pair discordance)
  p_min = 1 / (1 + Gamma)         lower bound

Sensitivity: report the Gamma at which the upper-bound p-value
first exceeds alpha.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import binom


def rosenbaum_pvalue(n_plus, n_minus, gamma):
    """Two-sided p at sensitivity parameter Gamma for matched pairs.

    n_plus  = # discordant pairs where treated has higher outcome (or is 'success')
    n_minus = # discordant pairs where control has higher outcome
    """
    n = n_plus + n_minus
    p_high = gamma / (1 + gamma)
    # Upper bound p = P(Binomial(n, p_high) >= n_plus) under one-sided (T > C)
    # For two-sided use max upper bound (either direction)
    p_upper = float(binom.sf(n_plus - 1, n, p_high))
    return p_upper


def sensitivity_curve(n_plus, n_minus, gammas=None, alpha=0.05):
    if gammas is None:
        gammas = np.arange(1.0, 5.01, 0.1)
    rows = []
    critical = None
    for g in gammas:
        p = rosenbaum_pvalue(n_plus, n_minus, g)
        rows.append({"gamma": float(g), "p_upper": p})
        if critical is None and p > alpha:
            critical = float(g)
    return {"rows": rows, "gamma_critical": critical}


if __name__ == "__main__":
    print("=== Rosenbaum bounds for matched-pairs sensitivity ===\n")
    # 60 discordant pairs; 45 favour treatment
    n_plus = 45; n_minus = 15
    r = sensitivity_curve(n_plus, n_minus)
    print(f"  Discordant pairs: n_plus = {n_plus}, n_minus = {n_minus}"
          f"   (favours treatment)")
    print(f"  Sensitivity curve (upper-bound p-value under Gamma):")
    for row in r["rows"][::4]:
        print(f"    Gamma = {row['gamma']:>4.1f}   p_upper = {row['p_upper']:.4f}")
    print(f"\n  Critical Gamma (p first exceeds 0.05): {r['gamma_critical']}")
    if r["gamma_critical"] is not None:
        print(f"  Interpretation: an unmeasured confounder ~{r['gamma_critical']}x more prevalent")
        print(f"  in one arm would be needed to overturn the study's conclusion.\n")

    print("--- library cross-check (R rbounds::binarysens, sensitivitymv; Python custom) ---")
