# Dispersion / Spread (Reference §1.2)

How spread out the data are around the center.

| Measure | Formula | Notes |
|---------|---------|-------|
| Range | `max - min` | Uses only the two extremes; very sensitive to outliers |
| Variance | `Σ(x - x̄)² / (n - 1)` | Sample variance (Bessel's correction, ddof=1); population uses `/ n` |
| Standard deviation | `√variance` | Same units as the data |
| IQR | `Q3 - Q1` | Robust; the spread of the middle 50% |
| Coefficient of variation | `sd / mean` | Unit-free relative spread — see `techniques/coefficient-of-variation` |
| Mean absolute deviation | `mean(\|x - center\|)` | `center` = mean or median |
| Median absolute deviation (MAD) | `median(\|x - median\|)` | Robust; ×1.4826 estimates σ under normality |

**Sample vs. population (ddof):** with a sample, divide variance by `n-1` (unbiased). With the full population, divide by `n`. numpy defaults to `ddof=0`; R's `var`/`sd` use `n-1`; here the from-scratch functions default to `ddof=1`.

## Files
- `python/dispersion.py` — from-scratch + numpy/scipy (`np.ptp`, `np.var(ddof=1)`, `scipy.stats.iqr/variation/median_abs_deviation`)
- `r/dispersion.R` — base R (`var`, `sd`, `range`, `IQR`, `mad`)
- `pyspark/dispersion.py` — Spark `var_samp`/`stddev_samp`; IQR & MAD via `approxQuantile`

## Run
```
python techniques/dispersion/python/dispersion.py
Rscript techniques/dispersion/r/dispersion.R
python techniques/dispersion/pyspark/dispersion.py
```

**Ref:** Devore, *Probability and Statistics for Engineering and the Sciences*, 9th ed., Cengage, 2016.
