# Standardized Coefficients & Dominance Analysis (Reference §5.38)

Two related answers to **"which predictor matters most?"** in a regression.

## Standardized coefficients ("beta weights")

Z-score `y` and each `xⱼ`, then refit OLS. The resulting coefficients are unit-free and on a comparable scale:

`β_j* = β_j · SD(xⱼ) / SD(y)`

`β_j* = 0.4` reads as "a 1-SD increase in `xⱼ` shifts `y` by 0.4 SD." That makes a coefficient on `income (USD)` comparable in magnitude to one on `age (years)` and to one on a `0/1` indicator.

## Dominance analysis (Budescu 1993)

Decomposes the model's `R²` into per-predictor **"general dominance"** contributions that **sum to total R²**. For each subset `S` not containing `xⱼ`, compute the gain `R²(S ∪ {j}) − R²(S)`; average over all `S` of each size; then average over all sizes:

`gen_dom_j = mean over size k of [ mean over S of size k not containing j of (R²(S∪{j}) − R²(S)) ]`

This is the **Shapley value** of `xⱼ` in the "cooperative game" where the payoff is the model's `R²`. It splits `R²` "fairly" across correlated predictors in a way that single-predictor `r²` or unique-contribution `R²` can't.

**Cost:** evaluates all `2^p` subsets. Feasible for `p ≲ 15`. For larger `p`, use **random subset sampling** (Shapley sampling) — still gives unbiased estimates.

## When the two disagree

A predictor can have a **large beta weight but small dominance** if much of its apparent effect is *shared* with correlated predictors. The beta weight is "the unique contribution given the rest of the model"; the dominance contribution is "the average across all sub-models, including those without the correlated rivals."

Demo (`p = 4` with `cor(x1, x2) = 0.7`):

```
beta-weight ranking : x1 > x3 > x2 > x4
dominance ranking   : x1 > x2 > x3 > x4   (x2 catches up; it's mostly captured by x1's beta but not by its dominance)
```

Report **both** in a "variable importance" section.

## Caveats

- **Standardization doesn't fix collinearity**, it just rescales it. Two highly correlated predictors still have unstable individual coefficients on the z-scored design.
- **Don't standardize binary predictors** if you want the coefficient to mean "the effect of being in the category" — z-scoring a 0/1 variable mangles its interpretation.
- **Dominance is descriptive of THIS model**, not causal. It splits R² across the predictors *as fit*; including a confounder shifts the decomposition without changing the underlying causal structure.

## Files
- `python/standardized_coefficients.py` — from-scratch beta weights (refit OLS on z-scored design) and exhaustive-subset dominance analysis (`2^p` fits); demos that the dominance contributions sum to the total `R²` exactly. Cross-checks against `statsmodels.OLS` on z-scored inputs.
- `r/standardized_coefficients.R` — from-scratch versions + `lm.beta::lm.beta` (standardized betas) and `relaimpo::calc.relimp(type = "lmg")` (the canonical implementation of general-dominance / Lindeman-Merenda-Gold).
- PySpark: N/A — both methods are post-fit transforms. For huge data, fit the OLS in MLlib (`techniques/multiple-linear-regression/pyspark/`) and compute beta weights from `coefficients × SD(xⱼ) / SD(y)` on the driver. Dominance analysis at Spark scale is prohibitively expensive — sample subsets if needed.

## Run
```
python techniques/standardized-coefficients/python/standardized_coefficients.py
Rscript techniques/standardized-coefficients/r/standardized_coefficients.R
```

**Refs:** Budescu, "Dominance Analysis: A New Approach to the Problem of Relative Importance of Predictors in Multiple Regression," *Psychological Bulletin* 114(3), 542–551, 1993; Azen & Budescu, "The Dominance Analysis Approach for Comparing Predictors in Multiple Regression," *Psychological Methods* 8(2), 129–148, 2003; Lindeman, Merenda & Gold, *Introduction to Bivariate and Multivariate Analysis*, Scott Foresman, 1980.
