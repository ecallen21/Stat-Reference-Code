# Weighted Least Squares (Reference §5.10)

When the errors are **heteroscedastic** — `Var(εᵢ) = σᵢ²` not constant — OLS is still unbiased but no longer efficient, and its standard errors are wrong.

**WLS** gives each observation a weight `wᵢ ∝ 1/σᵢ²`:

`β̂_WLS = (X' W X)⁻¹ X' W y`, where `W = diag(wᵢ)`.

Equivalent to running OLS on the **transformed** system `(W^½ X) β = W^½ y` — which is why the algorithm is essentially "scale each row by `√wᵢ`, then OLS."

## Choosing the weights

| Situation | Weights |
|-----------|---------|
| `Var(εᵢ)` known proportional to some function of `xᵢ` | `wᵢ = 1 / Var(εᵢ)` |
| Multiplicative error (`Var ∝ x²`) | `wᵢ = 1 / xᵢ²` |
| Poisson-like (`Var ∝ x`) | `wᵢ = 1 / xᵢ` |
| `yᵢ` is the **average** of `nᵢ` replicates | `wᵢ = nᵢ` |
| `yᵢ` is the **sum** of `nᵢ` replicates | `wᵢ = 1/nᵢ` |
| Unknown — estimate from the data | Iteratively reweighted (below) |

## Iteratively reweighted scheme (unknown weights)

1. Fit OLS, get residuals `eᵢ`.
2. Regress `log(eᵢ²)` on `X` to estimate `log Var(εᵢ)` (the **variance model**).
3. Set `wᵢ = 1/exp(X·γ̂)ᵢ`.
4. Refit WLS, repeat to convergence.

This is one of many possible variance-model parameterizations; if you have a strong prior on the form, plug that in instead.

## What WLS doesn't fix

WLS corrects **inefficiency** under heteroscedasticity, but only if your weights are right (or close). If you're not sure of the variance model:

- **Heteroscedasticity-robust standard errors** (White / HC0–HC3) keep OLS estimates and just fix the SEs. Use `cov_type = "HC3"` in statsmodels or `sandwich::vcovHC` in R. Less efficient than a correct WLS but **safer when the variance model is wrong**.
- For small `n`, HC3 is preferable to HC0/HC1.

## Files
- `python/weighted_least_squares.py` — from-scratch WLS via the `W^½` transformation; iteratively reweighted scheme for unknown weights; compares against `statsmodels.WLS`. Demo on `Var ∝ x²` data shows OLS / WLS / IRWLS coefficients converging and SEs shrinking with proper weighting.
- `r/weighted_least_squares.R` — from-scratch + base `lm(formula, weights = ...)`.
- PySpark: N/A — Spark's `LinearRegression` accepts a `weightCol`. For massive data with a known weights column, that's the direct route; for the IRWLS loop you'd repeat fit + variance-model-update.

## Run
```
python techniques/weighted-least-squares/python/weighted_least_squares.py
Rscript techniques/weighted-least-squares/r/weighted_least_squares.R
```

**Refs:** Aitken, "IV.—On Least Squares and Linear Combination of Observations," *Proc. Royal Soc. Edinburgh* 55, 42–48, 1935; Carroll & Ruppert, *Transformation and Weighting in Regression*, Chapman & Hall, 1988.
