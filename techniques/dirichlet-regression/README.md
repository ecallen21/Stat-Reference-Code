# Dirichlet Regression (Reference §47.324)

Campbell & Mosimann (1987); Maier (2014). Model COMPOSITIONAL
outcomes `y` on the simplex (y_j > 0, Σ y_j = 1) as
`Dir(α(X))` with `α_j = exp(xᵀ β_j)`:

```
y | X ~ Dir(α(X))
log α_j(x) = xᵀ β_j
```

Fit by MLE. Compositional outcomes arise in bioinformatics
(taxa proportions), budget shares, and election outcomes.

## Files

- `python/dirichlet_regression.py` — L-BFGS MLE for a
  3-category composition on 3 covariates, n=300. Estimated
  coefficient matrix closely tracks the truth; predicted
  mean composition at x=(1, 0.5, 0.5) is (0.60, 0.23, 0.17).
- `r/dirichlet_regression.R` — `DirichletReg`,
  `brms(family=dirichlet)` (R); `dirichlet` PyPI, PyMC /
  NumPyro, from-scratch (Python).

## When to use

- **Compositional outcomes** — proportions summing to 1.
- **Microbiome / metagenomics** — taxa relative abundances.
- **Budget shares / market shares** — where each row is a
  proportion vector.
- **Multinomial-like data with continuous proportions** —
  rather than counts.

## When NOT to use

- **Zeros in the composition** — Dirichlet has zero density
  at boundaries; use zero-inflated Dirichlet or additive
  log-ratio (ALR) transform + normal regression.
- **Very high K categories** — pairwise correlations
  restricted; consider Dirichlet-Multinomial or
  logistic-normal (Aitchison).
- **Random compositional effects across observations** —
  hierarchical Dirichlet or brms is preferred.

## Assumptions & caveats

- **Independence of composition components** given the
  covariates — Dirichlet enforces a specific negative-
  correlation structure; check with residual plots.
- **Log-link identifiability** — one α is often fixed
  (reference category) for interpretability.
- **Precision parameter** — the total sum φ = Σ α_j
  controls concentration; separate model for φ (Maier's
  DirichletReg "alternative" parameterisation) can help.
- **Aitchison alternatives** — log-ratio transforms
  (ALR, ILR, CLR) followed by multivariate normal regression
  offer flexible alternatives.

## Related in this repo

- `compositional-data` — CoDA framework.
- `multinomial-logistic` — categorical companion.
- `beta-regression` — the K=2 special case.
- `tweedie-glm-regression`, `inverse-gaussian-glm` — other
  exponential-family GLMs.
- `gamlss` — distributional-regression extension.

## Run

```
python techniques/dirichlet-regression/python/dirichlet_regression.py
Rscript techniques/dirichlet-regression/r/dirichlet_regression.R
```

**Refs:** Campbell, G. and Mosimann, J.E. "Multivariate methods for proportional shape." *ASA Proc. Sec. Statistical Graphics*, 1987; Maier, M.J. "DirichletReg: Dirichlet regression for compositional data in R." *J. Stat. Softw.*, 2014.

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
