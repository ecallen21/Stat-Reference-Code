# Collinearity Diagnostics (Reference §5.23)

**Collinearity** = strong linear relationships among the predictor columns of `X`. Effects: `Var(β̂)` explodes, individual coefficients become unstable and substantively wrong-signed, predictions are still fine but inference is unreliable.

## Diagnostics

| Diagnostic | Definition | Rule of thumb |
|------------|-----------|----------------|
| **Pairwise correlations** | `cor(xᵢ, xⱼ)` | Easy first pass; **misses joint** multicollinearity |
| **VIF (Variance Inflation Factor)** | `1 / (1 − R²_j)`, where `R²_j` regresses `xⱼ` on the *other* predictors | `> 5` noticeable, `> 10` serious |
| **Tolerance** | `1 / VIF_j` | `< 0.1` mirrors `VIF > 10` |
| **Condition index** | `√(λ_max / λ_k)` from singular values of the **scaled** `X` | `> 30` moderate-to-severe |
| **Variance-decomposition proportions** | for each near-singular dimension, how much of `Var(β̂_j)` it carries | Pairs of coefficients with `> 0.5` of variance on the same dim → collinear pair |

VIF only catches **one-other-predictor** redundancy. Three predictors that jointly add to a fourth pass VIF but show up clearly in the condition-index / variance-proportions view.

## What to do about it

(See §5.29 of the reference doc for the full list.)

1. **Drop one of a near-redundant pair** if substantively defensible.
2. **Center** before forming polynomial / interaction terms (`techniques/interaction-terms` `5.25`) — much of "collinearity" in those models is mechanical and disappears with centering.
3. **Regularize** with ridge / elastic net (`techniques/regularization`) — ridge specifically *solves* the collinearity problem by adding `λI` to `X'X`.
4. **PCR / PLS** (future batch) — replace the collinear group with a smaller set of orthogonal components.
5. If you only need **prediction**, collinearity barely matters — it inflates coefficient variance but not the fitted values.

## What it's not

- Collinearity is a property of the **design**, not the response. It doesn't bias `β̂`; it just inflates `Var(β̂)`.
- A "significant" model with no significant predictors is the classic signature of collinearity (high overall F, low individual t-stats).

## Files
- `python/collinearity_diagnostics.py` — from-scratch pairwise correlation matrix, VIF / tolerance, Belsley–Kuh–Welsch condition indices + variance-decomposition proportions; compares against `statsmodels.stats.outliers_influence.variance_inflation_factor`.
- `r/collinearity_diagnostics.R` — from-scratch versions + `car::vif`, `perturb::colldiag`.
- PySpark: N/A — these are properties of the design matrix; for huge data compute the predictor correlation matrix (`pyspark.ml.stat.Correlation`) on the cluster and run VIF on the small `p × p` summary on the driver.

## Run
```
python techniques/collinearity-diagnostics/python/collinearity_diagnostics.py
Rscript techniques/collinearity-diagnostics/r/collinearity_diagnostics.R
```

**Refs:** Belsley, Kuh & Welsch, *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*, Wiley, 1980; Marquardt, "Generalized Inverses, Ridge Regression, Biased Linear Estimation, and Nonlinear Estimation," *Technometrics* 12(3), 591–612, 1970.
