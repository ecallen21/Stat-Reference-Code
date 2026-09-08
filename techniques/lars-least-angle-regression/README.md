# LARS - Least Angle Regression (Reference §47.122)

Efron, Hastie, Johnstone & Tibshirani (2004). At each step, extend
the active set along the EQUIANGULAR direction (all active-column
signs equal) until a new predictor's absolute correlation with the
residual matches the current maximum:

    1. Initialise β=0, r=y.
    2. Add argmax|X_j'r| to A.
    3. Move β along direction equiangular to sign(X_j'r), j∈A,
       until |X_k'r| catches up for k∉A.
    4. Repeat.

A single-step modification recovers the full LASSO path (LARS-lasso).

## Files

- `python/lars_least_angle_regression.py` — from-scratch LARS
  with the equiangular direction + step-size computation. Demo
  (n=200, p=20, true support {0, 3, 7, 15} with coefs (1.5, −1.0,
  0.8, −0.6)):
  - Step 5 (all four true features active): β̂ = (1.66, −1.16, 0.71,
    −0.53)  ≈ truth.
  - Later steps overshoot toward OLS (use LARS-lasso stop rule
    in practice).
- `r/lars_least_angle_regression.R` — `lars`, `glmnet` (R);
  `sklearn.linear_model.Lars` / `lars_path`, from-scratch (Python).

## When to use

- **Full LASSO solution path** at cost of a single OLS.
- **Feature-selection ordering** — informative even when you
  ultimately use a different sparsity level.
- **Small-to-moderate p** where the equiangular step is affordable.
- **Analytic connection between forward-stagewise and LASSO** —
  educational value.

## When NOT to use

- **Very large p** — coordinate descent (`glmnet`) faster.
- **Correlated / group-lasso needs** — LARS doesn't handle groups
  natively; use LARS-EN or coord descent.
- **When only cross-validated λ is needed** — coord descent path +
  CV is standard.

## Assumptions & caveats

- **Standardise X** first — LARS is scale-sensitive.
- **Uniqueness** — solution unique if design columns are in
  general position.
- **LARS vs LASSO**: LARS-lasso mod adds a check that any active
  β_j crossing zero forces removal from A.
- **Numerical stability**: guard against near-singular Cholesky
  of active Gram matrix.

## Related in this repo

- `ridge-lasso-elasticnet`, `coordinate-descent-lasso`,
  `admm-consensus`, `adaptive-lasso`, `group-lasso`,
  `fused-lasso`, `scad-mcp-penalties`, `debiased-lasso`,
  `dantzig-selector`, `stability-selection` — sparse regression
  family.
- `orthogonal-matching-pursuit` — greedy alternative from the
  same era.
- `sure-independence-screening`, `model-x-knockoffs`,
  `post-selection-inference` — feature-selection cousins.

## Run

```
python techniques/lars-least-angle-regression/python/lars_least_angle_regression.py
Rscript techniques/lars-least-angle-regression/r/lars_least_angle_regression.R
```

**Refs:** Efron, B., Hastie, T., Johnstone, I. & Tibshirani, R. "Least angle regression." *Ann Stat* 32(2): 407-499, 2004.

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
