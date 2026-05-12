# Distribution Shape: Skewness & Kurtosis (Reference §1.4)

Quantify asymmetry and tail behavior.

## Skewness
`g1 = m3 / m2^(3/2)` where `m_k` is the k-th central moment.
- `g1 > 0` — right-skewed (long right tail), e.g. income
- `g1 < 0` — left-skewed (long left tail), e.g. age at death in developed countries
- `g1 ≈ 0` — symmetric

Variants: **biased** (method-of-moments) `g1`; **bias-corrected Fisher–Pearson** `G1 = g1·√(n(n-1))/(n-2)` (what most software reports); **Pearson's second** `3(mean - median)/sd`.

## Kurtosis
`g2 = m4 / m2² − 3` ("excess" kurtosis; normal = 0).
- `g2 > 0` — leptokurtic (heavy tails, peaked)
- `g2 < 0` — platykurtic (light tails, flat)

Bias-corrected: `G2 = ((n+1)g2 + 6)(n-1)/((n-2)(n-3))`. Watch out: some packages (R's `moments::kurtosis`) report **non-excess** kurtosis where normal = 3.

## Notes
- L-moments (L-skewness, L-kurtosis) are robust alternatives based on order statistics — preferred for small or heavy-tailed samples. See `techniques/l-moments`.
- Uses: checking the normality assumption, selecting a distribution, deciding whether a transformation is needed.

## Files
- `python/shape.py` — from-scratch + `scipy.stats.skew` / `scipy.stats.kurtosis` (with `bias=` toggle)
- `r/shape.R` — base R + `moments`, `e1071` (note `type=` argument), `psych::describe`
- PySpark: N/A (a single-pass moment scan; not a distributed-compute showcase — `skewness()`/`kurtosis()` aggregations exist in Spark SQL if you ever need them on huge data)

## Run
```
python techniques/shape-skewness-kurtosis/python/shape.py
Rscript techniques/shape-skewness-kurtosis/r/shape.R
```

**Refs:** Sheskin, *Handbook of Parametric and Nonparametric Statistical Procedures*, 5th ed., 2011; Hosking, "L-Moments," *JRSS-B* 52(1), 105–124, 1990.
