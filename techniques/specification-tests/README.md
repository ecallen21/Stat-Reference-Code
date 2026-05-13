# Specification & Diagnostic Tests for OLS (Reference §5.7, §5.21)

Formal hypothesis tests of the linear-model assumptions, built on top of a fitted OLS.

| Test | H₀ | What it actually checks |
|------|----|--------------------------|
| **Breusch–Pagan** | homoscedastic errors | Regress `eᵢ²` on the predictors `X`; `LM = n·R²_aux ~ χ²_{p−1}` |
| **White** | homoscedastic errors AND no omitted nonlinearity | BP on an expanded set: `X`, `X²`, all `xᵢ·xⱼ`; `LM = n·R²_aux ~ χ²_{q−1}` |
| **Durbin–Watson** | no first-order autocorrelation | `DW = Σ(eₜ − eₜ₋₁)² / Σeₜ²`; ≈ 2 → no AC, < 2 positive, > 2 negative |
| **Ramsey RESET** | correctly specified (no omitted nonlinear functions of `X`) | Add `ŷ²` (and optionally `ŷ³`) to the model; F-test their joint significance |
| **Shapiro–Wilk on residuals** | residuals normal | (see `techniques/normality-tests`) |
| **Jarque–Bera on residuals** | residuals normal | quick skew + kurt check (§5.21) |

## A practical workflow

1. **Plot first.** Residuals vs. fitted, scale-location, Q–Q of residuals, residuals vs. leverage (`techniques/regression-diagnostics`). Patterns usually jump out before any test does.
2. **Then test.** Use the formal tests to confirm what the plots already showed (or didn't).
3. **Match the test to the worry.**
   - Funnel-shaped residuals → **Breusch–Pagan** or **White**.
   - Time-ordered data with persistent residuals → **Durbin–Watson** (and consider AR / time-series models, Ch. 13).
   - Curved residuals vs. fitted → **RESET**, then add a polynomial / spline (`techniques/polynomial-regression`, `techniques/splines-segmented`).
   - Non-normal Q–Q → **Shapiro–Wilk** / **Jarque–Bera**; consider a transformation or robust regression (`techniques/robust-regression`).

## Watch out

- **Breusch–Pagan misses some heteroscedasticity** that doesn't depend *linearly* on the columns of `X`. The demo shows a case where Var(ε) ∝ |x₁| — BP passes, **White catches it**. White is the more general (but lower-power) check.
- **All these tests reject easily with large `n`.** A "significant" result on `n = 100,000` may correspond to a deviation too small to matter. Pair the test with an effect-size view (size of the heteroscedasticity, the polynomial coefficient, etc.).
- For inference under suspected heteroscedasticity, use **robust (White / HC) standard errors** directly — `cov_type = "HC3"` in statsmodels / `sandwich::vcovHC` in R — rather than testing first.

## Files
- `python/specification_tests.py` — from-scratch Breusch–Pagan, White, Durbin–Watson, Ramsey RESET, Shapiro–Wilk / Jarque–Bera on residuals; `run_all(X, y)` evaluates them all in one call. Demos clean / heteroscedastic / omitted-nonlinearity data. Cross-checks against `statsmodels.stats.diagnostic`.
- `r/specification_tests.R` — from-scratch versions + `lmtest::bptest`, `lmtest::dwtest`, `lmtest::resettest`; notes on `car::ncvTest`.
- PySpark: N/A — these are residual-based checks on a fitted small-`p` model.

## Run
```
python techniques/specification-tests/python/specification_tests.py
Rscript techniques/specification-tests/r/specification_tests.R
```

**Refs:** Breusch & Pagan, "A Simple Test for Heteroscedasticity and Random Coefficient Variation," *Econometrica* 47(5), 1287–1294, 1979; White, "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity," *Econometrica* 48(4), 817–838, 1980; Durbin & Watson, "Testing for Serial Correlation in Least Squares Regression," *Biometrika* 37–38, 1950–1951; Ramsey, "Tests for Specification Errors in Classical Linear Least-Squares Regression Analysis," *JRSS-B* 31(2), 350–371, 1969.
