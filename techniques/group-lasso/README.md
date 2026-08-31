# Group LASSO (Reference §32.9)

Yuan & Lin (2006). Selects **groups of features together** via an
L2-then-L1 penalty:

```
min_β  ½ ‖ y − X β ‖²  +  λ · Σ_g √|g| · ‖β_g‖_2.
```

Solved by group-wise soft thresholding (block coordinate descent):

```
β_g ← max(0, 1 − λ √|g| / ‖β_g^OLS‖) · β_g^OLS.
```

## Standard use cases

- **Dummy-coded factor levels** — the entire factor is in or out.
- **Spline basis functions per predictor** — smooth term is included
  or dropped as a unit.
- **Multi-response Gaussian regression** — rows of the coefficient
  matrix grouped.

## When to use

- **Structured sparsity** — group membership is known from the
  problem.
- **Categorical predictors with many levels** — avoids selecting only
  a subset of dummies.
- **Grouped genomics variables** — pathway-level selection.

## When NOT to use

- **Sparsity within groups** — plain LASSO or sparse-group-LASSO
  (Simon 2013).
- **Unknown group structure** — data-driven grouping first.

## Files

- `python/group_lasso.py` — from-scratch block-coordinate descent
  with per-group soft threshold. Demo 5 groups × 4 features, only
  groups 0 and 3 truly active. **Selected exactly groups 0, 3**;
  groups 1, 2, 4 zeroed out entirely; within-group coefficients
  recovered.
- `r/group_lasso.R` — `grplasso`, `gglasso` (R); `celer`,
  `group-lasso` (Python).

## Assumptions & caveats

- **λ scaling** — the `√|g|` weight equalises the penalty across
  differently-sized groups.
- **Overlapping groups** — Jacob-Obozinski-Vert 2009 overlap group
  LASSO handles them; block-CD is more subtle.
- **Sparse-group LASSO** (Simon 2013) adds within-group L1.
- **Standard errors** — sandwich / bootstrap; group-level selective
  inference is an open area.

## Related in this repo

- `ridge-lasso-elasticnet`, `adaptive-lasso`, `scad-mcp-penalties`,
  `fused-lasso`, `debiased-lasso`, `stability-selection`,
  `model-x-knockoffs` — the sparse-fitting family.
- `categorical-variable-coding` — the reason group structure often
  arises.
- `restricted-cubic-splines` — spline basis grouping.

## Run

```
python techniques/group-lasso/python/group_lasso.py
Rscript techniques/group-lasso/r/group_lasso.R
```

**Refs:** Yuan, M. & Lin, Y. "Model selection and estimation in regression with grouped variables." *JRSS-B*, 2006; Meier, L., van de Geer, S. & Bühlmann, P. "The group LASSO for logistic regression." *JRSS-B*, 2008; Simon, N. et al. "A sparse-group LASSO." *JCGS*, 2013.

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
