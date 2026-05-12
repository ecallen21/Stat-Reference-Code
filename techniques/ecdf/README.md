# Empirical Cumulative Distribution Function (Reference §1.13)

`F_n(t) = (# observations ≤ t) / n` — a right-continuous step function that jumps by `1/n` at each data point (more where there are ties). Properties:

- It is the **nonparametric MLE** of the true CDF `F`.
- `√n (F_n − F)` converges to a Brownian bridge (Donsker's theorem) — the basis of the Kolmogorov–Smirnov test.
- Underlies P–P plots and the (nonparametric) bootstrap (resampling = drawing from `F_n`).
- Its inverse is the sample quantile function (Hyndman–Fan type 1).

**DKW confidence band** (Dvoretzky–Kiefer–Wolfowitz): a simultaneous `1−α` band `F_n(t) ± ε` with `ε = √( ln(2/α) / (2n) )`, clipped to `[0,1]`. Unlike pointwise CIs, it covers the *whole* curve at once.

## Files
- `python/ecdf.py` — callable `ECDF` class (from scratch) with `.quantile()` and `.dkw_band()`; compares against `statsmodels` `ECDF` and `scipy.stats.ecdf`
- `r/ecdf.R` — from-scratch step function + DKW band; compares against base `stats::ecdf`
- `pyspark/ecdf.py` — ECDF as a `groupBy` + running-sum window; `evaluate()` looks up `F_n(t)` via a join

## Run
```
python techniques/ecdf/python/ecdf.py
Rscript techniques/ecdf/r/ecdf.R
python techniques/ecdf/pyspark/ecdf.py
```

**Refs:** Wilk & Gnanadesikan, "Probability Plotting Methods for the Analysis of Data," *Biometrika* 55(1), 1–17, 1968; Massart, "The Tight Constant in the DKW Inequality," *Ann. Probab.* 18(3), 1990.
