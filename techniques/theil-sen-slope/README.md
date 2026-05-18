# Theil–Sen Slope Estimator (Reference §6.32)

Robust **nonparametric** slope estimator for `y = β₀ + β₁x + ε`:

`β̂₁ = median { (yⱼ − yᵢ) / (xⱼ − xᵢ) : i < j }`
`β̂₀ = median { yᵢ − β̂₁ · xᵢ }`

The slope is the median of all `n(n−1)/2` pairwise slopes; the intercept is the median residual.

## Properties

- **29.3% breakdown** — up to ~29% of the data can be arbitrarily corrupted before the estimate breaks. (OLS: 0%.)
- **Distribution-free**: no normality assumption on `ε`.
- **Asymptotically normal**; CI on the slope comes from inverting Kendall's tau (Sen 1968) — the `k`-th smallest and `k`-th largest of the pairwise slopes, with `k` chosen via the normal-approximation to the Kendall null distribution.
- Slightly less efficient than OLS at the normal, but **much more robust** to outliers.

## When to reach for it

- Trend estimation in time-series environmental / hydrological data (the canonical use case).
- Robust slope when the residuals are heavy-tailed or contain outliers.
- Quick "is there a trend?" check that doesn't depend on shape.

For more than one predictor, see **rank-based regression** (§6.31, deferred) or **robust regression** (`techniques/robust-regression`).

## Cost

- `O(n²)` pairwise slopes; feasible for `n ≲ 10⁴`.
- For very large `n`, the **repeated-median** estimator (`mblm::mblm(..., repeated = TRUE)` in R) is `O(n² log n)` but more robust still.

## Files
- `python/theil_sen_slope.py` — from-scratch enumeration of all pairwise slopes, plus the Sen-1968 CI; demo on data with two extreme y's recovers slope 1.87 (true = 2.0) while OLS collapses to 1.17. Matches `scipy.stats.theilslopes`.
- `r/theil_sen_slope.R` — from-scratch + `mblm::mblm`.
- PySpark: N/A — `O(n²)` self-join is impractical; sample first.

## Run
```
python techniques/theil-sen-slope/python/theil_sen_slope.py
Rscript techniques/theil-sen-slope/r/theil_sen_slope.R
```

**Refs:** Theil, "A Rank-Invariant Method of Linear and Polynomial Regression Analysis," *Proc. Royal Netherlands Acad. Sci.* 53, 1950; Sen, "Estimates of the Regression Coefficient Based on Kendall's Tau," *JASA* 63(324), 1379–1389, 1968.
