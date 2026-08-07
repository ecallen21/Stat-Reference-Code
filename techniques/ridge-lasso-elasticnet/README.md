# Ridge, LASSO, and Elastic Net (Reference §5.9, §5.10)

Penalized regression that trades bias for variance reduction. All three shrink the OLS coefficients toward zero; they differ in the penalty geometry and hence in the sparsity pattern of the fitted coefficients.

## Ridge (Hoerl & Kennard 1970) — L2 penalty

```
minimize  ‖y − Xβ‖² + λ ‖β‖₂²
closed form:  β̂ = (XᵀX + λI)⁻¹ Xᵀy
```

Shrinks every coefficient smoothly; **never sets to exactly zero**. Best when many small-effect predictors matter and multicollinearity is severe.

## LASSO (Tibshirani 1996) — L1 penalty

```
minimize  ‖y − Xβ‖² + λ ‖β‖₁
```

Solved by **coordinate descent** (Friedman-Hastie-Tibshirani 2010). Sparsity: many coefficients become **exactly zero** → variable selection. Weak with highly-correlated predictor groups (picks one at random).

## Elastic Net (Zou & Hastie 2005) — mixed L1 + L2

```
minimize  (1/2) ‖y − Xβ‖² + λ [α ‖β‖₁ + (1 − α)/2 ‖β‖₂²]
```

`α ∈ [0, 1]` interpolates: `α = 0` → ridge, `α = 1` → LASSO. Middle values give sparsity **and** group-selection: correlated predictors enter or exit together.

## Regularization path

Fit at a log-spaced grid of `λ` from `λ_max` down to `λ_max / 1000`. Choose `λ` by cross-validation (see `cross-validation`). `λ_max` for standardized `X`: `λ_max = max_j |Xⱼᵀy| / (n · α)`.

## Files

- `python/ridge_lasso_elasticnet.py` — closed-form ridge + coordinate-descent Elastic Net (LASSO as `α = 1`, ridge as `α = 0`) + regularization-path grid. Demo (n = 200, p = 20, 5 true non-zeros): LASSO coefficients match `sklearn.linear_model.Lasso` to 3 decimals; path shows 0 → 8 → 20 non-zeros as `λ` decreases.
- `r/ridge_lasso_elasticnet.R` — `glmnet::glmnet` + `glmnet::cv.glmnet` (the canonical R implementation).

## When to use each

- **Ridge** — many small effects, no need for selection; high collinearity you want smoothed rather than resolved.
- **LASSO** — you believe most coefficients are zero and want a small, interpretable model.
- **Elastic Net** — high-dimensional (`p > n`), grouped predictors, or when LASSO gives unstable variable selection.

## Assumptions & caveats

- **Standardize `X`** before fitting — otherwise the penalty is scale-dependent.
- **Intercept unpenalized** — center `y` and standardize `X`, fit sans-intercept, then back-transform.
- **λ chosen by CV** — nested CV if you also need to report generalization error (see `nested-cv`).
- **Inference**: post-LASSO OLS or debiased LASSO for standard errors; the raw LASSO coefficients are biased.

## Run

```
python techniques/ridge-lasso-elasticnet/python/ridge_lasso_elasticnet.py
Rscript techniques/ridge-lasso-elasticnet/r/ridge_lasso_elasticnet.R
```

**Refs:** Hoerl, A.E. & Kennard, R.W. "Ridge regression: biased estimation for nonorthogonal problems." *Technometrics* 12(1), 55–67, 1970; Tibshirani, R. "Regression shrinkage and selection via the LASSO." *J. R. Stat. Soc. B* 58(1), 267–288, 1996; Zou, H. & Hastie, T. "Regularization and variable selection via the elastic net." *J. R. Stat. Soc. B* 67(2), 301–320, 2005; Friedman, J., Hastie, T. & Tibshirani, R. "Regularization paths for generalized linear models via coordinate descent." *J. Stat. Softw.* 33(1), 1–22, 2010.

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
