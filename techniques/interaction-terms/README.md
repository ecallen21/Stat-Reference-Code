# Interaction Terms in Regression (Reference §5.16, §5.24, §5.25, §5.37)

An **interaction** means the effect of one predictor depends on the level of another. You add a product column to the design matrix:

`y = β₀ + β₁ x₁ + β₂ x₂ + β₃ (x₁ · x₂) + ε`

The effect of `x₁` on `y` is then `β₁ + β₃ · x₂` — it *moves* with `x₂`. The same idea works for **categorical × continuous** (the slope on the continuous predictor differs between groups) and **categorical × categorical** (cell-specific means rather than additive row + column).

## Continuous × continuous: **center first** (§5.25)

`x₁`, `x₂`, and `x₁·x₂` are mechanically collinear when neither is centered — and that collinearity inflates the SEs on the *main effects* β₁ and β₂. Centering both predictors before forming the product:

- leaves the **fitted values, R², σ̂, and the interaction coefficient β₃ unchanged**;
- **reduces** the SEs on β₁ and β₂ dramatically (often 5–10×);
- makes the main effects interpretable as "the slope at the mean of the other variable" — which is usually the substantive question anyway.

Centering is not a magic fix for genuine collinearity, but for the *artificial* collinearity introduced by the product term it's essentially free.

## Categorical × continuous: simple slopes

For `y ~ x + group + x:group` with `group` dummy-coded (reference = group A):
- `β_x` is the slope **in the reference group**.
- `β_x + β_{x:group[k]}` is the slope **in group k**.
- A test on `β_{x:group[k]}` is "does group k's slope differ from group A's?"

The Python demo recovers `slope_by_group = {A: 0.2, B: 0.6, C: 1.0}` from the dummy-coded fit.

## Interpreting the coefficients (§5.24, §5.37)

- **β₁ alone is not "the effect of x₁."** It's the effect of x₁ *when x₂ = 0* (or when the reference category is selected). If 0 isn't in the data range, the number is uninterpretable without centering.
- After centering, β₁ is the effect of x₁ at the *mean* of x₂ — usually what you actually want to report.
- The marginal-effect helper `dE[y]/dx₁ = β₁ + β₃·x₂` is the right thing to evaluate at chosen `x₂` values (e.g. mean ± 1 SD) for an "effect at typical / high / low x₂" summary.

## When to suspect an interaction

- Theory or domain knowledge predicts moderation.
- A scatter plot of residuals vs. each predictor stratified by another shows different slopes.
- The marginal `R²` is unimpressive but you have reason to believe effects differ across subgroups.

Don't fish for interactions by testing all `p(p−1)/2` products without correction — that's a multiple-testing problem (`techniques/multiple-comparisons`).

## Files
- `python/interaction_terms.py` — from-scratch continuous×continuous (with centering toggle), categorical×continuous with simple-slope recovery, and a marginal-effect helper. Shows numerically that centering preserves R² and the interaction coefficient while collapsing main-effect SEs.
- `r/interaction_terms.R` — from-scratch + idiomatic `lm(y ~ x1 * x2)` and `lm(y2 ~ x * group)`.
- PySpark: N/A — once you've built the interaction column with `F.col("x1") * F.col("x2")`, hand it to `pyspark.ml.regression.LinearRegression`.

## Run
```
python techniques/interaction-terms/python/interaction_terms.py
Rscript techniques/interaction-terms/r/interaction_terms.R
```

**Refs:** Aiken & West, *Multiple Regression: Testing and Interpreting Interactions*, SAGE, 1991; Faraway, *Linear Models with R*, 2nd ed., 2014.
