# Tetrachoric & Polychoric Correlation (Reference §4.7)

Correlation between **ordinal** variables under the assumption that each one is a thresholded version of an unobserved continuous variable that's jointly **bivariate normal** with the other.

- **Tetrachoric** = 2 × 2 case (two binary variables).
- **Polychoric** = general r × c (two ordinal variables).

## Model

Unobserved `(X*, Y*) ~ N(0, 0, 1, 1, ρ)`. Thresholds `α_0 < α_1 < … < α_r` on `X*` and `β_0 < β_1 < … < β_c` on `Y*` map the latent draws to the observed ordinal categories:
`X = i ⟺ α_{i−1} < X* ≤ α_i`, similarly for `Y`.

## Estimation (two-step)

1. **Thresholds from marginals**: `α_i = Φ⁻¹(Σ_{k ≤ i} p_{k.})`, and similarly for `β_j`.
2. **ρ from the joint counts**: maximize the multinomial log-likelihood over `ρ ∈ (−1, 1)`, where each cell's probability is the bivariate-normal rectangle defined by its thresholds.

## When this matters

The plain Pearson correlation on coded ordinal data (1/2/3/4/5) systematically **underestimates** the underlying association — sometimes badly when only a few categories are used. Polychoric is the right tool for:
- Factor analysis / SEM on Likert items (heavily used in psychometrics).
- Ordinal-by-ordinal contingency tables in social science.

A widely cited correction: with 5+ near-symmetric categories, Pearson ≈ polychoric; with 3 or fewer, the gap can be > 0.10.

## SE / inference

Standard errors require the Hessian of the log-likelihood. We don't compute them here — for inference, use `polycor::polychor(..., std.err = TRUE)` in R or `psych::polychoric` (which also returns a correlation matrix for many ordinal items).

## Files
- `python/polychoric_correlation.py` — from-scratch threshold estimation + 1-D maximization of the multinomial log-likelihood via `scipy.optimize.minimize_scalar`; uses `scipy.stats.multivariate_normal.cdf` for the bivariate-normal rectangle probabilities. Demos 3×3 (polychoric) and 2×2 (tetrachoric).
- `r/polychoric_correlation.R` — from-scratch version (requires the `mvtnorm` package for the bivariate CDF) + `polycor::polychor` for the canonical implementation with SEs.
- PySpark: N/A — small contingency table once the joint counts are aggregated; reuse `techniques/chi-square-tests/pyspark/` for the counts.

## Run
```
python techniques/polychoric-correlation/python/polychoric_correlation.py
Rscript techniques/polychoric-correlation/r/polychoric_correlation.R
```

**Refs:** Pearson, "Mathematical Contributions to the Theory of Evolution. VII. On the Correlation of Characters Not Quantitatively Measurable," *Phil. Trans. Royal Soc. A* 195, 1–47, 1900; Olsson, "Maximum Likelihood Estimation of the Polychoric Correlation Coefficient," *Psychometrika* 44(4), 443–460, 1979.
