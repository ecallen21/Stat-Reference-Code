# Gini Coefficient & Lorenz Curve (Reference §1.23)

Measures of **inequality / concentration** in a non-negative distribution — income, wealth, healthcare utilization, biodiversity evenness, etc.

## Lorenz curve
Sort the values ascending. Plot cumulative *population* share `p_i = i/n` (x-axis) against cumulative *value* share `L_i = (sum of the i smallest values) / (sum of all)` (y-axis). The 45° line `L = p` is perfect equality; the curve sags below it as inequality grows.

## Gini coefficient
`G = 2 × (area between the equality line and the Lorenz curve)`.

Equivalent forms:
- **Trapezoidal**: `G = 1 − 2·∫ L dp` (area under the Lorenz curve via the trapezoid rule).
- **Mean absolute difference**: `G = Σᵢ Σⱼ |xᵢ − xⱼ| / (2 n² x̄)`; on sorted data this is the `O(n log n)` form `G = 2·Σᵢ i·x₍ᵢ₎ / (n·Σx) − (n+1)/n`.
- **Bias-corrected** (small samples): multiply by `n/(n−1)`.

Range: `0` (everyone equal) → `1` (one unit holds everything; the sample max with `n` units is `1 − 1/n`).

## Files
- `python/gini_lorenz.py` — from-scratch Lorenz curve + trapezoidal & mean-difference Gini (and bias-corrected); includes equality / max-inequality sanity checks. (numpy/scipy have no built-in Gini, so the "library" version is the canonical numpy one-liner.)
- `r/gini_lorenz.R` — from-scratch + `ineq::Gini` (`corr=TRUE` for bias correction) and `ineq::Lc`; notes on `DescTools::Gini`, `reldist::gini`
- `pyspark/gini_lorenz.py` — sorted ranks via a window function → Gini without collecting; Lorenz curve as a cumulative-share DataFrame

## Run
```
python techniques/gini-lorenz/python/gini_lorenz.py
Rscript techniques/gini-lorenz/r/gini_lorenz.R
python techniques/gini-lorenz/pyspark/gini_lorenz.py
```

**Ref:** Ceriani & Verme, "The Origins of the Gini Index," *Journal of Economic Inequality* 10(3), 421–443, 2012.
