# Multicollinearity — VIF & Condition Number (Reference §41.8)

Kutner-Nachtsheim-Neter-Li (2005), Dormann et al. (2013). Two
diagnostics for detecting collinear predictors that inflate SEs and
destabilise coefficients.

## Metrics

- **VIF** (variance inflation factor)
  `VIF_j = 1 / (1 − R²_j)` where `R²_j` is from regressing predictor
  `j` on all other predictors.
  - `> 5` — concern.
  - `> 10` — serious.
- **Condition number** = `√(λ_max / λ_min)` of `X^T X`
  (on standardised `X`).
  - `< 15` — fine.
  - `15–30` — moderate.
  - `> 30` — severe.

## Fixes

- **Drop / consolidate** highly-collinear predictors on domain
  grounds.
- **Centre before** creating interactions or polynomial terms.
- **Ridge regression** — regularisation robust to collinearity.
- **PCA / PLS** — reduce to independent components.

## When to use

- **Every multivariable OLS / GLM fit** before interpreting β SEs.
- **When Wald p-values seem "off"** — inflated SEs are the usual
  culprit.

## When NOT to use

- **Pure prediction models with ridge / LASSO** — the penalty
  handles collinearity implicitly; VIF is informative but not
  actionable.
- **Trees / random forests** — VIF is meaningless for tree-based
  models.

## Files

- `python/multicollinearity_vif.py` — VIF via regressing each
  predictor on the rest + condition number from SVD. Demo (n=300,
  `x1 ≈ x2` at r ≈ 1, `x3` independent): initial **VIF(x1) = 441,
  VIF(x2) = 441, condition # 42**; after dropping `x2`, all VIFs
  = 1.00, condition # 1.05.
- `r/multicollinearity_vif.R` — `car::vif`,
  `performance::check_collinearity`, `caret::findCorrelation`,
  `perturb::colldiag` (R); `statsmodels.variance_inflation_factor`,
  `numpy.linalg.cond` (Python).

## Assumptions & caveats

- **Numerical predictors** — for categorical, use generalised VIF
  (`car::vif` reports GVIF^(1/(2·df))).
- **Interaction terms** — always centre main effects first, else
  VIFs are spurious.
- **Perfect collinearity** crashes OLS with singular design; VIF
  reports infinity.
- **Threshold rules** are heuristics, not laws — a VIF of 8 in a
  small-effect estimate is different from a VIF of 8 in a large
  one.

## Related in this repo

- `penalized-clinical-prediction`, `ridge-regression` — the
  regularised alternative.
- `probabilistic-pca`, `random-projections` — feature reduction.
- `multivariable-model-building` — full-model strategy.

## Run

```
python techniques/multicollinearity-vif/python/multicollinearity_vif.py
Rscript techniques/multicollinearity-vif/r/multicollinearity_vif.R
```

**Refs:** Kutner, M.H., Nachtsheim, C.J., Neter, J., & Li, W. *Applied Linear Statistical Models*, 5th ed., McGraw-Hill, 2005 (ch 7, 10); Dormann, C.F. et al. "Collinearity: a review of methods to deal with it and a simulation study evaluating their performance." *Ecography*, 2013.

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
