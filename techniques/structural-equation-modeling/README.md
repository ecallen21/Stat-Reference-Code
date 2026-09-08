# Structural Equation Modeling (Reference §47.137)

Jöreskog (1970). Models latent constructs and their relations by
combining a **measurement model** (indicators load onto latent
factors) with a **structural model** (latent regressions among
factors). Parameters fit by minimising the LISREL ML discrepancy:

    F_ML(theta) = log|Sigma(theta)| + tr(S Sigma^-1) - log|S| - p.

Widely used in psychology, education, epidemiology and clinical
outcomes research for confirmatory factor analysis (CFA), mediation,
and multi-group invariance testing.

## Files

- `python/structural_equation_modeling.py` — one-factor CFA (p = 5
  indicators, n = 400 simulated). ML-fit converges in ~150 Nelder-
  Mead iters; recovered lambda ≈ [0.85, 0.78, 0.73, 0.51, 0.45] vs
  truth [0.90, 0.80, 0.70, 0.60, 0.50]; chi² ≈ 7.3 with df = 5, CFI
  ≈ 0.985.
- `r/structural_equation_modeling.R` — canonical `lavaan::cfa`
  workflow with fit indices + standardised loadings.

## When to use

- **Latent constructs** with multiple imperfect indicators (quality
  of life, patient-reported outcomes, personality traits).
- **Mediation / path analysis** with measurement error corrected.
- **Multi-group invariance** testing (configural / metric / scalar).

## When NOT to use

- **Single-indicator regression** — plain regression is enough.
- **Non-normal ordinal indicators** without WLSMV estimator.
- **Small n** (< ~200) with many parameters — biased SE, weak fit.

## Assumptions & caveats

- **Multivariate normality** of indicators (or use robust ML).
- **Identifiability** requires fixing scale (e.g. `std.lv=TRUE`).
- **Modification indices** are exploratory; over-fitting inflates
  fit but reduces cross-validity.

## Related in this repo

- `factor-analysis`, `principal-component-analysis` — measurement-
  side ancestors.
- `path-analysis`, `mediation-analysis` — structural-side siblings.
- `latent-class-analysis`, `mixture-models` — categorical latent
  analogues.
- `multilevel-models` — combine with SEM for MSEM.

## Run

```
python techniques/structural-equation-modeling/python/structural_equation_modeling.py
Rscript techniques/structural-equation-modeling/r/structural_equation_modeling.R
```

**Refs:** Jöreskog, K. G. "A general method for analysis of covariance structures." *Biometrika* 57(2), 1970; Bollen, K. *Structural Equations with Latent Variables*, Wiley, 1989; Rosseel, Y. "lavaan: An R package for structural equation modeling." *J Stat Softw* 48(2), 2012.

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
