# Coordinate Descent Lasso (Reference §47.67)

Friedman, Hastie & Tibshirani (2010). Cycle j = 1, …, p and
update

    β_j ← S(X_j' r_partial / n , λ) / (X_j' X_j / n)

where `r_partial = y − Xβ + X_jβ_j` and
`S(z, γ) = sign(z)·max(|z|−γ, 0)` is the soft-threshold. With warm
starts, strong-rule screening, and covariance updates this powers
**`glmnet`** — full λ-path in seconds.

## Files

- `python/coordinate_descent_lasso.py` — coordinate descent for
  lasso from scratch. Demo (n=200, p=20, true support
  {0, 3, 7, 15}):
  - λ=0.05  converges in 6 sweeps, exact support recovery,
    coefs (1.39, −0.96, 0.64, −0.59) vs truth (1.5, −1.0, 0.8, −0.6)
  - larger λ → stronger shrinkage of true nonzero coefficients.
- `r/coordinate_descent_lasso.R` — `glmnet` (R);
  `sklearn.linear_model.Lasso`, `celer`, `skglm`, from-scratch
  (Python).

## When to use

- **Sparse regression / classification** in low- to
  moderate-dimensional p (up to millions with strong-rule screening).
- **Full regularisation path** — CV / IC-based λ selection.
- **GLM extensions** — Poisson, logistic, Cox all fit via IRLS +
  coord descent (`glmnet`).
- **Grouped / hierarchical penalties** — extends to group lasso
  by block-wise coordinate descent.

## When NOT to use

- **Non-convex penalties** (SCAD, MCP) — need difference-of-convex
  or LLA majorisation, then coord descent (`ncvreg`).
- **Very ill-conditioned design** — proximal-gradient / ADMM may
  converge faster.
- **Prediction only, no sparsity** — ridge or ElasticNet is often
  fine.

## Assumptions & caveats

- **Standardisation** — assumes columns of X are standardised;
  restore original scale afterwards.
- **Uniqueness** — lasso solution not unique if columns of X are
  collinear.
- **Active set** — use strong rules or KKT screening to skip
  guaranteed-zero coordinates each iteration.
- **λ path** — descend from λ_max = ‖X'y‖_∞ / n on a log grid;
  warm-start next λ from previous solution.

## Related in this repo

- `ridge-lasso-elasticnet` — the reference article family.
- `admm-consensus` — alternative solver for the same problem.
- `adaptive-lasso`, `scad-mcp-penalties`, `debiased-lasso`,
  `stability-selection`, `group-lasso`, `fused-lasso`,
  `dantzig-selector` — penalised-regression cousins.
- `bagging-oob`, `stability-selection`,
  `sure-independence-screening` — model-selection stabilisation.

## Run

```
python techniques/coordinate-descent-lasso/python/coordinate_descent_lasso.py
Rscript techniques/coordinate-descent-lasso/r/coordinate_descent_lasso.R
```

**Refs:** Friedman, J., Hastie, T. & Tibshirani, R. "Regularization paths for generalized linear models via coordinate descent." *JSS* 33(1): 1-22, 2010; Tibshirani, R. et al. "Strong rules for discarding predictors in lasso-type problems." *JRSS-B* 74(2): 245-266, 2012.

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
