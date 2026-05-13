# Polynomial Regression (Reference §5.3)

Add curvature to a linear model by including powers of a predictor.

`y = β₀ + β₁ x + β₂ x² + … + β_d xᵈ + ε`

Still **linear in the parameters** β — so it's just OLS on a transformed design matrix.

## Raw vs. orthogonal basis

| Basis | Columns | Pros | Cons |
|-------|---------|------|------|
| **Raw** | `[1, x, x², …, xᵈ]` | Coefficients have direct interpretation | Columns are **highly correlated** → VIF explodes, individual `t`-stats are unreliable, coefficients flip wildly as you add a degree |
| **Orthogonal** | Gram–Schmidt-orthonormalized columns | Coefficients are **uncorrelated**; the `t`-test on `c_k` is the test for the degree-`k` contribution above and beyond lower degrees | Coefficients don't read off as "the effect of `xᵏ`" — the basis isn't the powers |

**Fitted values, R², σ̂ are identical** between the two bases — they span the same column space. Only the *parameterization* (and hence the individual coefficients / their tests) differs.

## Choosing the degree

The right way: fit increasing degrees on the **orthogonal** basis, stop when the highest-order coefficient is no longer significant. The example does exactly this and recovers degree 2 on data generated as `y = 1 + 0.5x + 0.8x² + ε`.

Why orthogonal: with the raw basis, the `t`-test on `β_3` in a degree-3 fit is "the unique contribution of `x³` given `1, x, x²` are already in" — but those columns are so correlated that the test has very low power. The orthogonal basis fixes that.

## Practical caveats

- **Degree ≥ 4 is rarely right.** Polynomials oscillate badly at the edges of the data (Runge phenomenon). Use **splines** instead (`techniques/splines-segmented`).
- Always plot the fitted curve to see what you're committing to.
- **Centering** `x` before squaring helps with collinearity in raw basis (covered in `techniques/interaction-terms` `§5.25`).
- For prediction outside the observed range, the polynomial can diverge — never trust extrapolation (§5.33).

## Files
- `python/polynomial_regression.py` — from-scratch raw + Gram–Schmidt orthogonal bases, OLS fit with full coefficient table, `choose_degree` helper using the orthogonal-basis `t`-tests; compares against `statsmodels.OLS`.
- `r/polynomial_regression.R` — from-scratch versions + base `poly(x, degree, raw = TRUE/FALSE)` inside `lm`.
- PySpark: N/A — once you've decided on the columns, the fit is just multiple regression (`techniques/multiple-linear-regression/pyspark/`).

## Run
```
python techniques/polynomial-regression/python/polynomial_regression.py
Rscript techniques/polynomial-regression/r/polynomial_regression.R
```

**Refs:** Faraway, *Linear Models with R*, 2nd ed., 2014; Harrell, *Regression Modeling Strategies*, 2nd ed., 2015 (Ch. 2 — argues hard against polynomials and for splines).
