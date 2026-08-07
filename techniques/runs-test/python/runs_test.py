"""Wald-Wolfowitz runs test for randomness (Reference §7.15).

A RUN is a maximal subsequence of identical elements.  Given a binary sequence
with n_1 A's and n_2 B's, count the total number of runs R.  Under the null
(sequence is a random ordering) R has known mean and variance:

    mu_R = 2 n_1 n_2 / (n_1 + n_2) + 1
    sigma_R^2 = 2 n_1 n_2 (2 n_1 n_2 - n_1 - n_2) /
                 ((n_1 + n_2)^2 (n_1 + n_2 - 1))
    z = (R - mu_R) / sigma_R   ~ N(0, 1) asymptotically

Two-sided p tests for either TOO FEW runs (clustering: streaks) or TOO MANY
runs (over-alternation, rare).

Extension to continuous data
    Dichotomize around the median -> runs on above/below labels.  Useful for
    testing whether residuals of a regression fit are randomly scattered.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def count_runs(seq) -> int:
    seq = np.asarray(seq)
    if len(seq) == 0: return 0
    return int(1 + np.sum(np.asarray(seq[1:]) != np.asarray(seq[:-1])))


def wald_wolfowitz_runs(seq) -> dict:
    """Runs test on a binary or dichotomizable sequence."""
    s = np.asarray(seq)
    labels = np.unique(s)
    if len(labels) != 2:
        # Try dichotomizing at the median
        med = float(np.median(s))
        s = (s > med).astype(int)
        note = f"dichotomized at median ({med:.3f})"
        labels = np.array([0, 1])
    else:
        note = "already binary"
    n1 = int(np.sum(s == labels[0])); n2 = int(np.sum(s == labels[1]))
    R = count_runs(s)
    if n1 == 0 or n2 == 0:
        return {"error": "sequence has only one distinct value"}
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (R - mu) / math.sqrt(var)
    return {"n_runs": int(R), "n1": n1, "n2": n2,
            "mean_under_H0": float(mu),
            "sd_under_H0": float(math.sqrt(var)),
            "z": float(z),
            "p_two_sided": float(2 * stats.norm.sf(abs(z))),
            "note": note,
            "method": "Wald-Wolfowitz runs test"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Random binary sequence ===")
    seq = rng.choice([0, 1], size=100)
    r = wald_wolfowitz_runs(seq)
    print(f"  runs = {r['n_runs']}, expected under H0 = {r['mean_under_H0']:.1f}, z = {r['z']:.3f}, p = {r['p_two_sided']:.4f}")

    print("\n=== Clustered sequence (all 0s then all 1s) ===")
    seq = np.concatenate([np.zeros(50), np.ones(50)])
    r = wald_wolfowitz_runs(seq)
    print(f"  runs = {r['n_runs']}, expected = {r['mean_under_H0']:.1f}, z = {r['z']:.3f}, p = {r['p_two_sided']:.4f}")

    print("\n=== Alternating sequence (0101...) ===")
    seq = np.arange(100) % 2
    r = wald_wolfowitz_runs(seq)
    print(f"  runs = {r['n_runs']}, expected = {r['mean_under_H0']:.1f}, z = {r['z']:.3f}, p = {r['p_two_sided']:.4f}")

    print("\n=== Continuous residuals dichotomized at median ===")
    resid = rng.normal(size=80) + 0.05 * np.arange(80)  # slight trend
    r = wald_wolfowitz_runs(resid)
    print(f"  runs = {r['n_runs']}, z = {r['z']:.3f}, p = {r['p_two_sided']:.4f}   ({r['note']})")

    print("\n--- library cross-check (statsmodels runstest_1samp) ---")
    try:
        from statsmodels.sandbox.stats.runs import runstest_1samp
        z, p = runstest_1samp(np.arange(100) % 2, cutoff=0.5)
        print(f"  statsmodels alternating: z = {z:.3f}, p = {p:.4f}")
    except Exception as ex:
        print(f"  (statsmodels unavailable: {ex})")
