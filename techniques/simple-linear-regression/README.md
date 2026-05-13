# Simple Linear Regression (Reference §5.1)

Model: `yᵢ = β₀ + β₁ xᵢ + εᵢ`, with `εᵢ ~ iid N(0, σ²)`.

## Closed-form OLS

| Quantity | Formula |
|----------|---------|
| `β̂₁` | `Σ(x − x̄)(y − ȳ) / Σ(x − x̄)²` = `cov(x,y) / var(x)` |
| `β̂₀` | `ȳ − β̂₁ · x̄` |
| `σ̂²` | `RSS / (n − 2)` |
| `SE(β̂₁)` | `σ̂ / √Sₓₓ` |
| `SE(β̂₀)` | `σ̂ · √(1/n + x̄²/Sₓₓ)` |
| `R²` | `1 − RSS/TSS` |

(`Sₓₓ = Σ(x − x̄)²`)

## Intervals at a new `x₀`

| Kind | Variance term | Use |
|------|---------------|-----|
| **Confidence** (CI on the mean response) | `σ̂² · (1/n + (x₀ − x̄)²/Sₓₓ)` | "Average y at x₀" |
| **Prediction** (PI for a new observation) | `σ̂² · (1 + 1/n + (x₀ − x̄)²/Sₓₓ)` | "A single new y at x₀" |

The PI is wider — the `+1` is the irreducible noise around the line. A common bug is reporting a CI when a PI is what the question asks for.

## Assumptions

1. **Linearity** — `E[y | x]` is linear in `x`.
2. **Independence** — `εᵢ` are independent.
3. **Homoscedasticity** — `Var(εᵢ) = σ²` doesn't depend on `x`.
4. **Normality** of residuals (for inference; OLS estimates remain unbiased without it).
5. **No influential outliers**.

Diagnose with `techniques/regression-diagnostics`. For violations: WLS (`techniques/weighted-least-squares`), robust regression (`techniques/robust-regression`), or transformations.

## Files
- `python/simple_linear_regression.py` — from-scratch coefficients, SEs, t-tests, R², confidence and prediction intervals at new `x₀`; compares against `statsmodels.OLS`.
- `r/simple_linear_regression.R` — from-scratch + `stats::lm` / `summary(lm)` / `predict(... interval = "confidence" / "prediction")`.
- `pyspark/simple_linear_regression.py` — two routes: closed-form via sufficient statistics (`F.count`, `F.mean`, `F.sum(x*x)`, etc.) and `pyspark.ml.regression.LinearRegression`.

## Run
```
python techniques/simple-linear-regression/python/simple_linear_regression.py
Rscript techniques/simple-linear-regression/r/simple_linear_regression.R
python techniques/simple-linear-regression/pyspark/simple_linear_regression.py
```

**Refs:** Weisberg, *Applied Linear Regression*, 4th ed., Wiley, 2014; Kutner, Nachtsheim, Neter & Li, *Applied Linear Statistical Models*, 5th ed., McGraw-Hill, 2005.
