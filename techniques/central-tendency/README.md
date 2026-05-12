# Central Tendency (Reference §1.1)

Summaries of the "center" of a distribution.

| Estimator | Formula | Notes |
|-----------|---------|-------|
| Arithmetic mean | `sum(x) / n` | Sensitive to outliers; needs interval/ratio scale |
| Median | middle order statistic | Robust to outliers & skew; valid for ordinal data |
| Mode | most frequent value(s) | Only measure valid for nominal data; may not be unique |
| Trimmed mean | drop k% from each tail, average the rest | Compromise between mean and median robustness |
| Winsorized mean | replace each tail with its boundary value, then average | Like trimming but keeps n; pairs with Winsorized variance for inference |
| Weighted mean | `sum(w·x) / sum(w)` | Observations contribute unequally (survey weights, meta-analysis) |
| Geometric mean | `exp(mean(log x))`, x > 0 | Multiplicative processes: growth rates, fold-changes, antibody titers |
| Harmonic mean | `n / sum(1/x)`, x > 0 | Averaging rates and ratios (e.g. speeds) |

**When to use which:** symmetric, no outliers → mean; skewed or outliers → median or trimmed/Winsorized mean; nominal data → mode; ratios of a fixed quantity → harmonic; multiplicative/log-scale data → geometric.

## Files
- `python/central_tendency.py` — from-scratch + numpy/scipy (`np.mean`, `np.median`, `scipy.stats.mode/trim_mean/gmean/hmean`)
- `r/central_tendency.R` — base R (`mean`, `median`, `mean(trim=)`, `weighted.mean`) + `psych` (`geometric.mean`, `harmonic.mean`, `winsor.mean`)
- `pyspark/central_tendency.py` — Spark aggregations; median via `approxQuantile`

## Run
```
python techniques/central-tendency/python/central_tendency.py
Rscript techniques/central-tendency/r/central_tendency.R
python techniques/central-tendency/pyspark/central_tendency.py
```

**Ref:** Wackerly, Mendenhall & Scheaffer, *Mathematical Statistics with Applications*, 7th ed., Cengage, 2008.
