# Robust Location & Scale Estimators (Reference §1.3, §1.26)

Estimators that stay reliable when the data contain outliers or have heavy tails.

## Location
| Estimator | Idea | Notes |
|-----------|------|-------|
| Median | middle order statistic | 50% breakdown point; loses efficiency vs. mean for clean normal data |
| Trimmed mean | discard k% from each tail, average the rest | 10–20% trimming handles most situations; allows SE estimation (unlike the median) |
| Winsorized mean | clamp each tail to its boundary value, then average | Keeps n; pairs with Winsorized variance for inference |
| Huber M-estimator | IRLS: quadratic loss near center, linear in tails | Tuning constant k (default 1.345 ≈ 95% efficiency at the normal) |

## Scale
| Estimator | Formula | Notes |
|-----------|---------|-------|
| IQR | `Q3 - Q1` | Simple, robust |
| MAD | `median(\|x - median\|)` | ×1.4826 → consistent estimate of σ under normality |
| Winsorized variance | variance of the Winsorized sample | Used by Yuen's test |

## Two-sample comparison
**Yuen's trimmed t-test** — Welch-style t-test on *trimmed means* with *Winsorized variances*. A robust drop-in for the two-sample t-test when groups are skewed/heavy-tailed. `df` is estimated Welch-style.

## Files
- `python/robust_location_scale.py` — from-scratch + scipy (`median_abs_deviation`, `trim_mean`) and `statsmodels.robust.scale.huber`
- `r/robust_location_scale.R` — base R (`median`, `mad`, `mean(trim=)`) + `MASS::huber`, `WRS2::yuen`
- PySpark: N/A (these are small-sample robustness tools; not a distributed-compute use case)

## Run
```
python techniques/robust-location-scale/python/robust_location_scale.py
Rscript techniques/robust-location-scale/r/robust_location_scale.R
```

**Refs:** Wilcox, *Introduction to Robust Estimation and Hypothesis Testing*, 5th ed., 2022; Yuen, "The Two-Sample Trimmed t for Unequal Population Variances," *Biometrika* 61(1), 165–170, 1974; Huber & Ronchetti, *Robust Statistics*, 2nd ed., Wiley, 2009.
