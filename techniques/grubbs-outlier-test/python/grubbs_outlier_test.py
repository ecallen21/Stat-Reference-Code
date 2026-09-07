"""Grubbs' outlier test (Reference Sec 3.26).

Grubbs 1950 'Sample criteria for testing outlying observations',
Ann Math Stat. Formal one-sided or two-sided test that the MOST
EXTREME observation is an outlier under a Gaussian null.

Two-sided statistic:
    G = max_i |x_i - x_bar| / s

Reject H0 at level alpha if:
    G > (n - 1) / sqrt(n) * sqrt(t^2 / (n - 2 + t^2))
where t = t_{alpha / (2n), n - 2}.

Only one outlier tested per call; APPLY ITERATIVELY (removing the
extreme and re-testing) with Bonferroni-adjusted alpha, or use the
Rosner ESD test for multiple outliers.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import t as tdist


def grubbs_two_sided(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean(); sd = x.std(ddof=1)
    G = np.max(np.abs(x - mean)) / sd
    tcrit = tdist.ppf(1 - alpha / (2 * n), df=n - 2)
    Gcrit = (n - 1) / np.sqrt(n) * np.sqrt(tcrit ** 2 / (n - 2 + tcrit ** 2))
    return {"G": float(G), "G_crit": float(Gcrit), "reject": bool(G > Gcrit),
            "outlier_index": int(np.argmax(np.abs(x - mean))),
            "outlier_value": float(x[int(np.argmax(np.abs(x - mean)))])}


def rosner_esd(x, k=5, alpha=0.05):
    """Rosner Extreme Studentized Deviate test for up to k outliers."""
    x = list(np.asarray(x, dtype=float))
    orig = x[:]
    Rs = []; lambdas = []; removed = []
    for i in range(k):
        n = len(x)
        if n < 3: break
        mean = np.mean(x); sd = np.std(x, ddof=1)
        R = np.max(np.abs(np.array(x) - mean)) / sd
        idx = int(np.argmax(np.abs(np.array(x) - mean)))
        p = 1 - alpha / (2 * (n - i))
        tcrit = tdist.ppf(p, df=n - 2)
        lam = (n - 1) * tcrit / np.sqrt((n - 2 + tcrit ** 2) * n)
        Rs.append(R); lambdas.append(lam); removed.append(x[idx])
        del x[idx]
    #  Number of outliers = largest i such that R_i > lambda_i
    n_out = 0
    for i in range(len(Rs) - 1, -1, -1):
        if Rs[i] > lambdas[i]:
            n_out = i + 1; break
    return {"n_outliers": n_out, "R_stats": Rs, "critical": lambdas,
            "removed_values": removed}


if __name__ == "__main__":
    print("=== Grubbs' outlier test + Rosner ESD ===\n")
    rng = np.random.default_rng(0)
    #  Clean-ish sample with one obvious outlier
    x = np.r_[rng.normal(10, 1, 30), 20.0]
    r = grubbs_two_sided(x, alpha=0.05)
    print(f"  Single-outlier Grubbs:")
    print(f"    G = {r['G']:.3f}   G_crit = {r['G_crit']:.3f}   "
          f"reject H0 = {r['reject']}")
    print(f"    Flagged x[{r['outlier_index']}] = {r['outlier_value']:.2f}")

    #  Rosner test with 3 injected outliers
    x2 = np.r_[rng.normal(10, 1, 30), 20.0, 21.0, 22.0]
    r2 = rosner_esd(x2, k=5, alpha=0.05)
    print(f"\n  Rosner ESD (up to k = 5 outliers):")
    print(f"    Detected {r2['n_outliers']} outliers")
    print(f"    R stats: {[round(v, 2) for v in r2['R_stats']]}")
    print(f"    Critical: {[round(v, 2) for v in r2['critical']]}")

    print("\n--- library cross-check (outliers::grubbs.test / dixon.test R; scipy custom Python) ---")
