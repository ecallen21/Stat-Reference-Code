# Quantiles, Percentiles & Order Statistics (Reference §1.5)

The p-th quantile splits the data so that a fraction `p` lies at or below it. There is **no single definition** — Hyndman & Fan (1996) describe nine. The common ones:

| HF type | h (the "index") | Interpolation | Used by |
|---------|-----------------|---------------|---------|
| 1 | `n·p`, take `x_⌈h⌉` | none (step function = inverse empirical CDF) | discrete/exact-rank settings |
| 6 | `(n+1)·p − 1` | linear | Minitab, SPSS ("Weibull") |
| 7 | `(n−1)·p` | linear | **R's default**, numpy, pandas |

Other items here: **five-number summary** (min, Q1, median, Q3, max), **percentile rank** of a value, and a small **empirical CDF** helper (see `techniques/ecdf` for more).

> R's `fivenum()` uses *Tukey hinges*, which can differ slightly from the type-7 Q1/Q3 — both are "correct," just different conventions.

## Files
- `python/quantiles.py` — from-scratch HF types 1/6/7 + `np.quantile(method=...)` and `scipy.stats.scoreatpercentile` / `percentileofscore`
- `r/quantiles.R` — from-scratch + base `quantile(type=1..9)`, `fivenum`, `ecdf`
- `pyspark/quantiles.py` — `DataFrame.approxQuantile` (Greenwald-Khanna sketch) and `percentile_approx` inside `groupBy`

## Run
```
python techniques/quantiles/python/quantiles.py
Rscript techniques/quantiles/r/quantiles.R
python techniques/quantiles/pyspark/quantiles.py
```

**Refs:** Hyndman & Fan, "Sample Quantiles in Statistical Packages," *The American Statistician* 50(4), 361–365, 1996; David & Nagaraja, *Order Statistics*, 3rd ed., Wiley, 2003.
