# Pearson Correlation (Reference §4.1)

The most common measure of **linear** association between two continuous variables.

`r = Σ(x − x̄)(y − ȳ) / √(Σ(x − x̄)² · Σ(y − ȳ)²) = cov(x, y) / (sd(x) · sd(y))`

Range: −1 (perfect negative linear) … 0 (no linear) … +1 (perfect positive linear). `r²` is the proportion of variance in one variable "explained" by a linear fit to the other.

## Significance test and CI

Test `H₀: ρ = 0`:
`t = r · √(n − 2) / √(1 − r²)` ~ `t_{n−2}`.

CI on ρ via the **Fisher z transformation**:
- `z = atanh(r)`, `SE(z) = 1/√(n − 3)`
- CI on `z`: `z ± z_{α/2}·SE(z)`, then back-transform with `tanh`.

This is preferable to a Wald CI on `r` directly because the sampling distribution of `r` is skewed near `±1`; `atanh` stabilizes it.

## Assumptions / caveats

- **Linear** association only. A strong U-shape can have `r = 0` — always plot.
- **Sensitive to outliers**; a single extreme point can dominate the result.
- Significance test assumes approximate bivariate normality; for heavy tails or ordinal data prefer **Spearman** or **Kendall**.
- Correlation ≠ causation. Confounding is the default explanation, not the exception.

## Conventional benchmarks (Cohen 1988)

`|r|`: 0.1 = small, 0.3 = medium, 0.5 = large. Always context-dependent — see `techniques/effect-sizes` for the longer discussion.

## Files
- `python/pearson_correlation.py` — from-scratch `r`, t-test, Fisher z CI; compares against `scipy.stats.pearsonr` (which now also returns a CI). Includes the "r = 0 despite a clear relationship" demo with a quadratic.
- `r/pearson_correlation.R` — from-scratch + `stats::cor` and `stats::cor.test(method = "pearson")`.
- `pyspark/pearson_correlation.py` — Spark's built-in `F.corr` aggregation + closed-form Fisher-z CI on the driver; helper for a multi-column correlation list.

## Run
```
python techniques/pearson-correlation/python/pearson_correlation.py
Rscript techniques/pearson-correlation/r/pearson_correlation.R
python techniques/pearson-correlation/pyspark/pearson_correlation.py
```

**Refs:** Pearson, "Notes on Regression and Inheritance in the Case of Two Parents," *Proc. Royal Soc.* 58, 240–242, 1895; Fisher, "On the 'Probable Error' of a Coefficient of Correlation Deduced from a Small Sample," *Metron* 1, 3–32, 1921.
