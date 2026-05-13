# Multiple-Comparison Corrections (Reference §3.13, §3.14)

When you run `m` hypothesis tests in a single study, the chance of *at least one* false positive grows fast (`1 − (1−α)^m`). These procedures adjust the per-test p-values so a downstream "reject if p ≤ α" decision controls a chosen family-wise error rate.

## What they control

| Procedure | Controls | Assumption | Notes |
|-----------|----------|------------|-------|
| **Bonferroni** | FWER | none | `p_adj = min(1, m·pᵢ)`; simplest, most conservative |
| **Šidák** | FWER | independence | `1 − (1−pᵢ)ᵐ`; tiny gain over Bonferroni |
| **Holm** | FWER | none | Step-down; **uniformly more powerful than Bonferroni** — always prefer it over plain Bonferroni |
| **Hochberg** | FWER | independence / positive dep. | Step-up; more powerful than Holm under its assumption |
| **Benjamini–Hochberg (BH / FDR)** | FDR | independence / positive dep. | The standard for many-test settings (genomics, fMRI) |
| **Benjamini–Yekutieli (BY)** | FDR | any dependence | BH × `H_m` (harmonic number); much more conservative |

**FWER** = probability of *any* false rejection. **FDR** = expected proportion of false rejections among rejections (so a few false positives are OK if you find many true ones too).

## Choosing a procedure

- A handful of comparisons (e.g. 3–10 a-priori contrasts) and you want a strong "almost certainly no false positives" guarantee → **Holm**.
- All-pairs after ANOVA → use **Tukey HSD** (or **Games–Howell**) in `techniques/post-hoc-tests` — those bake the correction into the critical value, more powerful than applying these procedures to pairwise t-tests.
- Thousands of tests, "find true signals, tolerate some false ones" → **Benjamini–Hochberg**.
- Tests that may be negatively correlated → **Benjamini–Yekutieli**.

## Reading the output

Each procedure returns an adjusted p-value `p_adj` aligned with the original input order. The decision rule is the same in every case: **reject `H₀ᵢ` iff `p_adj_i ≤ α`**. You don't compare to `α/m` anymore — the adjustment is already in the number.

## When *not* to adjust

Pre-planned, theory-driven comparisons that you would test even in isolation generally don't need correction (some authors disagree). Exploratory testing on many endpoints always does. The reference doc's §3.47 has a longer discussion.

## Files
- `python/multiple_comparisons.py` — from-scratch step-down (Holm) and step-up (Hochberg, BH, BY) algorithms; compares against `statsmodels.stats.multitest.multipletests` (which calls Hochberg "simes-hochberg").
- `r/multiple_comparisons.R` — from-scratch + base `stats::p.adjust` (method strings: `"bonferroni"`, `"holm"`, `"hochberg"`, `"hommel"`, `"BH"` / `"fdr"`, `"BY"`).
- PySpark: N/A — these are pure post-processing on a small vector of p-values.

## Run
```
python techniques/multiple-comparisons/python/multiple_comparisons.py
Rscript techniques/multiple-comparisons/r/multiple_comparisons.R
```

**Refs:** Holm, "A Simple Sequentially Rejective Multiple Test Procedure," *Scandinavian J. Statistics* 6, 65–70, 1979; Hochberg, "A Sharper Bonferroni Procedure for Multiple Tests of Significance," *Biometrika* 75, 800–802, 1988; Benjamini & Hochberg, "Controlling the False Discovery Rate," *JRSS-B* 57(1), 289–300, 1995; Benjamini & Yekutieli, "The Control of the False Discovery Rate in Multiple Testing Under Dependency," *Ann. Statistics* 29(4), 1165–1188, 2001.
