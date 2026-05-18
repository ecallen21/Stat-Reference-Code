# Multiple Linear Regression (Reference §5.2)

Model: `y = Xβ + ε`, `ε ~ N(0, σ² Iₙ)`, where `X` is the `n × p` design matrix (first column = 1s for the intercept).

## Closed-form OLS

`β̂ = (X'X)⁻¹ X'y`,  `Var(β̂) = σ²(X'X)⁻¹`,  `σ̂² = RSS / (n − p)`.

Solve via QR / `np.linalg.lstsq` (not by inverting `X'X` directly) for numerical stability.

## Per-coefficient output

| Quantity | Formula |
|----------|---------|
| `SE(β̂_j)` | `σ̂ · √[(X'X)⁻¹]_{jj}` |
| `t_j` | `β̂_j / SE(β̂_j)` ~ `t_{n−p}` |
| 95% CI | `β̂_j ± t_{0.975, n−p} · SE(β̂_j)` |

## Overall fit

| Quantity | Formula | Notes |
|----------|---------|-------|
| `R²` | `1 − RSS/TSS` | Proportion of variance explained |
| **Adjusted `R²`** | `1 − (1 − R²)·(n−1)/(n−p)` | Penalizes adding predictors; the one to compare across models |
| **Overall F** | `(R² / (p−1)) / ((1−R²) / (n−p))` ~ `F(p−1, n−p)` | Tests `H₀: β₁ = … = β_{p−1} = 0` |

## Intervals at new design rows

For a new row `x₀`:
- Mean response CI: `Var = x₀' V x₀` (V = `Var(β̂)`)
- New-observation PI: `Var = x₀' V x₀ + σ̂²`
- Half-width: `t_{1−α/2, n−p} · √Var`

## What this technique sets up

This is the workhorse the rest of Batch 4 builds on:
- `regression-diagnostics` — residual plots, Cook's D, leverage, DFFITS / DFBETAS on a fitted multiple regression.
- `specification-tests` — Breusch–Pagan, Durbin–Watson, White, RESET.
- `collinearity-diagnostics` — VIF, condition index.
- `polynomial-regression`, `interaction-terms`, `splines-segmented` — all build their design matrices and hand them to OLS.
- `regularization` — adds `λ‖β‖²` (ridge) or `λ‖β‖₁` (lasso) to the OLS objective.
- `weighted-least-squares` — `β̂ = (X' W X)⁻¹ X' W y`.

## Files
- `python/multiple_linear_regression.py` — from-scratch OLS via `np.linalg.lstsq`; full coefficient table with SEs / t / p / 95% CI; R², adj R², overall F-test; CI/PI at new rows; compares against `statsmodels.OLS`.
- `r/multiple_linear_regression.R` — from-scratch via `solve(crossprod(X), crossprod(X, y))` + `stats::lm` / `summary` / `confint`.
- `pyspark/multiple_linear_regression.py` — two routes: sufficient-stats `X'X` and `X'y` aggregated in one Spark pass (then solve on the driver), and `pyspark.ml.regression.LinearRegression`.

## Run
```
python techniques/multiple-linear-regression/python/multiple_linear_regression.py
Rscript techniques/multiple-linear-regression/r/multiple_linear_regression.R
python techniques/multiple-linear-regression/pyspark/multiple_linear_regression.py
```

**Refs:** Weisberg, *Applied Linear Regression*, 4th ed., 2014; Faraway, *Linear Models with R*, 2nd ed., 2014.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
