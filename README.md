# Stat-Reference-Code

Working implementations of the techniques catalogued in
`stat_techniques_reference_v124.docx`, written in **R**, **Python**, and
**PySpark** (the latter only where distributed/large-data execution is
meaningful).

## Layout

Organized by technique, then language:

```
techniques/
  <technique-name>/
    README.md                # what it is, when to use it, formulas, assumptions, ref
    r/<technique>.R          # base-R "from scratch" + idiomatic-package version
    python/<technique>.py    # numpy/scipy "from scratch" + library version; example under __main__
    pyspark/<technique>.py   # Spark DataFrame version (when applicable)
```

Each implementation provides a *from-scratch* version (so the math is visible)
**and** an idiomatic library version (what you'd actually use), plus a small
runnable example.

## Running the examples

- Python: `pip install -r requirements.txt` then `python techniques/<name>/python/<name>.py`
- R: `Rscript techniques/<name>/r/<name>.R`
- PySpark: `python techniques/<name>/pyspark/<name>.py` (needs a local Spark install)

## Notation & Conventions

These names mean the same thing in every file unless a function's own docstring overrides them.

### Data inputs
| Name | Meaning |
|------|---------|
| `x`, `x1`, `x2` | A 1-D sample (a Python sequence / numpy array / R numeric vector). `x1` and `x2` are two independent samples (e.g. group 1 and group 2). |
| `y` | A second variable paired with `x` (same length), for bivariate techniques. Not used in Batch 1. |
| `w` | Weights, one per observation in `x` (`len(w) == len(x)`). Used by `weighted_mean`. Survey weights, meta-analysis weights, etc. |
| `df` *(PySpark)* | A Spark `DataFrame`. |
| `col`, `row_col`, `col_col`, `value_col`, `weight_col`, `group_col` *(PySpark)* | Column **names** (strings) inside `df`. |
| `groups` | A list/list-of-lists, one sub-sample per group (used by one-way ANOVA effect sizes). |
| `subjects` | A list of repeated-measurements vectors, one per subject (used by within-subject CV). |
| `events`, `person_time_total` | Event count and total person-time at risk (for incidence rates). |
| `x` / `n` *(proportions)* | Number of successes / sample size, in functions that take a count rather than a vector. |

### Parameters & options
| Name | Meaning |
|------|---------|
| `n` | Sample size (`len(x)`). |
| `proportion` | Fraction trimmed/Winsorized from **each** tail. `0.2` means drop the bottom 20% and the top 20%. Must be in `[0, 0.5)`. |
| `ddof` | "Delta degrees of freedom" — divisor is `n − ddof`. `ddof=1` is the sample (Bessel-corrected) version; `ddof=0` is the population version. |
| `conf` | Confidence level for a CI (e.g. `0.95`). |
| `alpha` | Significance level / `1 − conf`. |
| `bias` *(skew/kurtosis)* | `True` → method-of-moments estimator; `False` → bias-corrected `G1` / `G2` that most software reports. |
| `excess` *(kurtosis)* | `True` → normal distribution has kurtosis 0; `False` → normal has kurtosis 3. |
| `kind` *(quantile)* | Hyndman–Fan quantile definition (1, 6, or 7 — see `techniques/quantiles`). |
| `as_percent` *(CV)* | If `True`, return `100 × SD/mean` instead of `SD/mean`. |
| `relative_error` *(PySpark)* | Tolerance for `approxQuantile` (`0` = exact, expensive; `0.001` is a good default). |
| `bias_corrected` *(Gini)* | Multiply by `n/(n−1)` for the small-sample correction. |
| `max_iter`, `tol` *(Huber)* | IRLS stopping criteria. |
| `k` *(Huber)* | Tuning constant for the Huber loss; `1.345` gives ~95% efficiency at the normal. |

### Conventions in the code
- Python's from-scratch functions default to **`ddof = 1`** (sample variance / SD). numpy defaults to `ddof = 0`, so we pass `ddof=1` explicitly when comparing.
- R's `var()` / `sd()` use `n − 1` by default; both languages therefore agree on the from-scratch defaults.
- Functions that need strictly positive inputs (geometric/harmonic mean, geometric CV, log-based things) raise `ValueError` (Python) / `stop()` (R) if a non-positive value is passed.
- "From-scratch" implementations exist for **transparency** (you can see the formula); for production code prefer the library version shown alongside.
- Library cross-checks use optional packages (`pingouin`, `lmoments3`, R's `effsize` / `lmom` / `ineq` / `MASS` / `psych` / `DescTools`); each file degrades to a note rather than erroring if the package is missing.

## Progress

Building in batches; we walk through each batch together before moving on.

### Batch 1 — Chapter 1: Descriptive Statistics

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [central-tendency](techniques/central-tendency) | 1.1 | ✅ | ✅ | ✅ |
| 2 | [dispersion](techniques/dispersion) | 1.2 | ✅ | ✅ | ✅ |
| 3 | [robust-location-scale](techniques/robust-location-scale) | 1.3, 1.26 | ✅ | ✅ | N/A |
| 4 | [shape-skewness-kurtosis](techniques/shape-skewness-kurtosis) | 1.4 | ✅ | ✅ | N/A |
| 5 | [quantiles](techniques/quantiles) | 1.5 | ✅ | ✅ | ✅ |
| 6 | [ecdf](techniques/ecdf) | 1.13 | ✅ | ✅ | ✅ |
| 7 | [effect-sizes](techniques/effect-sizes) | 1.6, 1.25 | ✅ | ✅ | N/A |
| 8 | [frequency-crosstab](techniques/frequency-crosstab) | 1.7 | ✅ | ✅ | ✅ |
| 9 | [rates-proportions](techniques/rates-proportions) | 1.8 | ✅ | ✅ | ✅ |
| 10 | [coefficient-of-variation](techniques/coefficient-of-variation) | 1.22, 1.33 | ✅ | ✅ | ✅ |
| 11 | [gini-lorenz](techniques/gini-lorenz) | 1.23 | ✅ | ✅ | ✅ |
| 12 | [l-moments](techniques/l-moments) | 1.24 | ✅ | ✅ | N/A |

Later batches will cover the remaining chapters (Probability, Inference,
Correlation, Regression, GLMs, ...).
