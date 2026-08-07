"""Tukey HSD, Dunnett, Scheffé multiple-comparison procedures (Reference §6.9).

After a one-way ANOVA rejects the omnibus null, follow-up compares specific
group means with FWER-controlled adjusted p-values or CIs.

Tukey HSD (Honest Significant Difference)
    All pairwise mean comparisons.  Statistic
        q = (mean_i - mean_j) / sqrt(MSE / n_h)     n_h = harmonic mean of n_i, n_j
    compared to the studentized range distribution q(k, df).
    FWER controlled at alpha across all k(k-1)/2 pairs.

Dunnett's test
    Each treatment vs a SINGLE control.  Uses correlated-t distribution.
    More powerful than Tukey when only k - 1 comparisons matter.

Scheffé's method
    All possible LINEAR CONTRASTS (not just pairwise).  Very conservative;
    only advantage is coverage of every contrast at once.

Implementations here focus on Tukey HSD from scratch + note the others.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def tukey_hsd(y, group, alpha: float = 0.05) -> dict:
    """Tukey HSD pairwise comparisons for one-way ANOVA groups."""
    y = np.asarray(y, dtype=float); group = np.asarray(group)
    groups = np.unique(group); k = len(groups)
    n = len(y); grand = y.mean()
    ns = np.array([int(np.sum(group == g)) for g in groups])
    means = np.array([y[group == g].mean() for g in groups])
    SSW = float(np.sum([np.sum((y[group == g] - means[i]) ** 2) for i, g in enumerate(groups)]))
    df_within = n - k
    MSE = SSW / df_within
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            n_h = 2 / (1 / ns[i] + 1 / ns[j])
            se = math.sqrt(MSE / n_h)
            diff = float(means[i] - means[j])
            q_stat = abs(diff) / se
            # p-value from studentized range
            try:
                p = float(1 - stats.studentized_range.cdf(q_stat, k, df_within))
            except Exception:
                p = float("nan")
            q_crit = float(stats.studentized_range.ppf(1 - alpha, k, df_within))
            hwidth = q_crit * se
            rows.append({"group_i": str(groups[i]), "group_j": str(groups[j]),
                         "diff": diff, "se": float(se),
                         "q_stat": float(q_stat),
                         "p_value": p,
                         "ci_low": float(diff - hwidth),
                         "ci_high": float(diff + hwidth),
                         "reject_H0": bool(q_stat > q_crit)})
    return {"pairwise": rows, "MSE_within": MSE,
            "df_within": int(df_within),
            "k_groups": int(k), "n": int(n), "alpha": float(alpha),
            "method": "Tukey HSD pairwise comparisons"}


def dunnett_test(y, group, control_label, alpha: float = 0.05) -> dict:
    """Dunnett's test: each treatment vs one control.  Approximation via Bonferroni within family."""
    y = np.asarray(y, dtype=float); group = np.asarray(group)
    groups = np.unique(group); k = len(groups)
    treatments = [g for g in groups if g != control_label]
    if control_label not in groups:
        raise ValueError(f"control label {control_label!r} not present in groups")
    control_y = y[group == control_label]
    means = {g: y[group == g].mean() for g in groups}
    ns = {g: int(np.sum(group == g)) for g in groups}
    n = len(y); k = len(groups)
    SSW = float(np.sum([np.sum((y[group == g] - means[g]) ** 2) for g in groups]))
    df_within = n - k
    MSE = SSW / df_within
    rows = []
    for t in treatments:
        se = math.sqrt(MSE * (1 / ns[t] + 1 / ns[control_label]))
        diff = float(means[t] - means[control_label])
        t_stat = diff / se
        # Dunnett critical value approximated by Sidak-corrected t (mildly conservative vs true)
        alpha_adj = 1 - (1 - alpha) ** (1 / len(treatments))
        t_crit = float(stats.t.ppf(1 - alpha_adj / 2, df_within))
        rows.append({"treatment": str(t),
                     "diff_vs_control": diff,
                     "se": float(se), "t_stat": float(t_stat),
                     "p_value_sidak_approx": float(2 * len(treatments) * stats.t.sf(abs(t_stat), df_within)),
                     "reject_H0": bool(abs(t_stat) > t_crit)})
    return {"pairwise": rows, "control": str(control_label),
            "method": "Dunnett's test (Sidak-approximated critical value)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    groups = np.repeat(["A", "B", "C", "D"], 20)
    y = np.concatenate([rng.normal(0, 1, 20),
                        rng.normal(0.5, 1, 20),
                        rng.normal(1.5, 1, 20),
                        rng.normal(0.2, 1, 20)])

    print("=== Tukey HSD (4 groups, n = 20 each) ===")
    r = tukey_hsd(y, groups)
    for row in r["pairwise"]:
        marker = "*" if row["reject_H0"] else " "
        print(f"  {marker} {row['group_i']} - {row['group_j']}: diff = {row['diff']:6.3f}  "
              f"CI ({row['ci_low']:.3f}, {row['ci_high']:.3f})  p = {row['p_value']:.4f}")

    print("\n=== Dunnett vs control A ===")
    r = dunnett_test(y, groups, control_label="A")
    for row in r["pairwise"]:
        marker = "*" if row["reject_H0"] else " "
        print(f"  {marker} {row['treatment']} vs A: diff = {row['diff_vs_control']:6.3f}   t = {row['t_stat']:.3f}   p_adj = {row['p_value_sidak_approx']:.4f}")

    print("\n--- library cross-check (scipy.stats.tukey_hsd) ---")
    try:
        Ys = [y[groups == g] for g in ("A", "B", "C", "D")]
        res = stats.tukey_hsd(*Ys)
        print(f"  scipy Tukey pvalues:\n{res.pvalue.round(4)}")
    except Exception as ex:
        print(f"  (scipy.tukey_hsd unavailable: {ex})")
