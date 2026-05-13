# Post-Hoc Pairwise Comparisons (Reference §3.10, §3.11, §3.16)

After a significant one-way ANOVA, post-hoc tests pinpoint **which** pairs of group means differ — while controlling the family-wise error rate (FWER) for the set of comparisons.

| Test | Comparisons | Variance assumption | Use when |
|------|-------------|---------------------|----------|
| **Tukey HSD** / **Tukey–Kramer** | all `k(k−1)/2` pairs | equal | Equal variances; balanced (Tukey) or unequal `n` (Tukey–Kramer) |
| **Dunnett** | `k − 1` (each treatment vs. one control) | equal | One control group, several treatments |
| **Games–Howell** | all pairs | unequal | Unequal variances (the Welch analogue of Tukey) |
| **Scheffé** (Ch. 3.12, not implemented) | any contrast | equal | Need to test arbitrary contrasts, not just pairs |

## Key idea

Each test uses a reference distribution that builds the multiple-comparison correction **into** the critical value:

- Tukey & Games–Howell → **studentized range** `q(k, df)` (peak-of-k-means distribution).
- Dunnett → **multivariate t** with equicorrelation (depending on the design).

So the per-pair p-values these procedures report are *already* family-wise adjusted. You don't apply Bonferroni on top.

## When `n` differs across groups

Tukey–Kramer adjusts the SE to `√(MS_w · ½ · (1/nᵢ + 1/nⱼ))` rather than `√(2·MS_w/n)`. The scipy / R built-ins do this automatically.

## What about the equal-variance assumption?

ANOVA, Tukey HSD, and Dunnett all assume equal within-group variances. Check with Levene's or Brown–Forsythe (`techniques/homogeneity-of-variance`). If violated:

- All pairs → **Games–Howell**.
- Treatment vs. control → **Dunnett T3** or **Tamhane T2** (see §3.37; not in this file — `PMCMRplus` in R, `scikit-posthocs` in Python).

For **pre-planned** contrasts (a small handful of hypotheses you decided on before looking at the data), see `techniques/multiple-comparisons` — Bonferroni / Holm / Hochberg are appropriate and often more powerful than the all-pairs procedures here.

## Files
- `python/post_hoc_tests.py` — from-scratch Tukey HSD (with Kramer adjustment), Dunnett (`scipy.stats.dunnett` when available; Bonferroni-t fallback otherwise), Games–Howell; compares against `scipy.stats.tukey_hsd`.
- `r/post_hoc_tests.R` — from-scratch versions + `stats::TukeyHSD`, `multcomp::glht` for Dunnett; notes on `PMCMRplus::gamesHowellTest`.
- PySpark: N/A — post-hoc tests run on the small `k × k` table of group means / SEs; once the per-group sufficient statistics are aggregated (see `techniques/one-way-anova/pyspark/`), the post-hoc step is driver-side.

## Run
```
python techniques/post-hoc-tests/python/post_hoc_tests.py
Rscript techniques/post-hoc-tests/r/post_hoc_tests.R
```

**Refs:** Tukey, "The Problem of Multiple Comparisons," 1953 (unpublished); Kramer, "Extension of Multiple Range Tests to Group Means with Unequal Numbers of Replications," *Biometrics* 12, 307–310, 1956; Dunnett, "A Multiple Comparison Procedure for Comparing Several Treatments with a Control," *JASA* 50, 1096–1121, 1955; Games & Howell, "Pairwise Multiple Comparison Procedures with Unequal N's and/or Variances: A Monte Carlo Study," *J. Educational Statistics* 1, 113–125, 1976.
