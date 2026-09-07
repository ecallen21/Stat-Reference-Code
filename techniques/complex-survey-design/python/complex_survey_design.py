"""Complex survey design analysis (Reference Sec 27.7).

Cochran 1977 'Sampling Techniques'; Kish 1965 'Survey Sampling'.
Design-based inference from stratified, clustered, unequal-probability
surveys. Key building blocks:

    * SAMPLING WEIGHTS w_i = 1 / pi_i  (inverse inclusion probability)
    * HORVITZ-THOMPSON estimator of the total:
        T_HT = sum_i w_i * y_i
    * MEAN via ratio estimator:
        y_bar_HT = T_HT / N_HT
    * VARIANCE via linearisation OR Taylor / Fay's balanced repeated
      replicate (BRR) OR JACKKNIFE (delete-1 cluster) OR BOOTSTRAP
      (Rao-Wu / Rescaled bootstrap for cluster designs).

We demonstrate:
    * HT mean estimator with weights.
    * Stratified variance via the sum-of-stratum-variances rule.
    * Delete-1-cluster jackknife for a complex (stratified + PSU) design.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def horvitz_thompson_mean(y, w):
    """Weighted (Hajek) mean."""
    return float(np.sum(w * y) / np.sum(w))


def stratified_variance(y, w, strata):
    """Estimate Var(y_bar_HT) via sum of within-stratum variances (SRS-like)."""
    W = np.sum(w)
    var = 0.0
    for h in np.unique(strata):
        idx = strata == h
        n_h = idx.sum()
        if n_h < 2:
            continue
        w_h = np.sum(w[idx])
        y_h = np.average(y[idx], weights=w[idx])
        s2_h = np.sum(w[idx] * (y[idx] - y_h) ** 2) / (w_h - 1)
        var += (w_h / W) ** 2 * s2_h / n_h
    return float(var)


def jackknife_cluster(y, w, strata, psu):
    """Delete-1 PSU jackknife (within-stratum). Returns SE of the weighted mean."""
    theta_hat = horvitz_thompson_mean(y, w)
    reps = []
    for h in np.unique(strata):
        mask_h = strata == h
        psu_h = np.unique(psu[mask_h])
        n_h = len(psu_h)
        if n_h < 2:
            continue
        for p in psu_h:
            drop = mask_h & (psu == p)
            #  Scale up remaining PSUs' weights within stratum h by n_h / (n_h - 1)
            w_rep = w.copy()
            w_rep[drop] = 0
            w_rep[mask_h & ~drop] *= n_h / (n_h - 1)
            reps.append(horvitz_thompson_mean(y, w_rep))
    reps = np.asarray(reps)
    #  Rao-Wu factor:  var = sum_h (n_h - 1) / n_h * sum_{PSU in h} (theta_rep - theta_full)^2
    var = 0.0; idx0 = 0
    for h in np.unique(strata):
        mask_h = strata == h
        psu_h = np.unique(psu[mask_h])
        n_h = len(psu_h)
        if n_h < 2:
            continue
        seg = reps[idx0: idx0 + n_h]
        var += (n_h - 1) / n_h * float(np.sum((seg - theta_hat) ** 2))
        idx0 += n_h
    return theta_hat, float(np.sqrt(var))


if __name__ == "__main__":
    print("=== Complex survey design analysis ===\n")
    rng = np.random.default_rng(0)
    #  4 strata, each with 5 PSUs, each PSU has 20 respondents (n = 400).
    H = 4; nPSU = 5; nSSU = 20; n = H * nPSU * nSSU
    strata = np.repeat(np.arange(H), nPSU * nSSU)
    psu = np.tile(np.repeat(np.arange(nPSU), nSSU), H) + strata * nPSU
    #  Unequal weights: strata h have weight (h + 1) * baseline
    w_baseline = 100.0
    w = np.array([(strata[i] + 1) * w_baseline for i in range(n)])
    #  Outcome with stratum-level shift
    y = strata * 2.0 + rng.normal(scale=1.0, size=n)

    #  Compare naive (unweighted) vs HT-weighted mean
    y_naive = float(np.mean(y))
    y_ht = horvitz_thompson_mean(y, w)
    print(f"  Naive mean (ignores weights)   = {y_naive:.3f}")
    print(f"  Horvitz-Thompson weighted mean = {y_ht:.3f}")
    print(f"  (Weights shift the estimate because heavier strata have y bigger.)")

    #  Stratified SE
    se_strat = np.sqrt(stratified_variance(y, w, strata))
    print(f"\n  Stratified linearisation SE    = {se_strat:.4f}")

    theta_jk, se_jk = jackknife_cluster(y, w, strata, psu)
    print(f"  Delete-1 PSU jackknife SE      = {se_jk:.4f}")

    print("\n--- library cross-check (survey R; samplics / statsmodels-survey Python) ---")
