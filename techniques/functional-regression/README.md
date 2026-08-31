# Functional Regression (Reference §31.3)

**Scalar-on-function regression**: predict a scalar `y` from a curve
`X(t)`:

```
y_i  =  α  +  ∫ X_i(t) · β(t) dt  +  ε_i
```

Two practical fitting routes:

- **FPC regression** — expand `X_i` in top-K functional principal
  components; regress `y` on the FPC scores.
- **Basis expansion + roughness penalty** — write `β(t) = Σ_j c_j
  φ_j(t)`, penalise second-derivative of `β`.

## When to use

- **Predict outcomes from time-series / spectra / growth curves** —
  medical waveform → risk score, near-infrared spectrum → chemical
  concentration.
- **Interpretable coefficient function** `β(t)` shows which parts of
  the curve matter.

## When NOT to use

- **Curves are best represented pointwise** (long, jagged, non-smooth)
  — use flexible ML with the raw time series.
- **Function-on-function regression** — needs a bilinear kernel; see
  `refund::pffr`.
- **Nonlinear effects** — see FANOVA or single-index / additive
  functional models.

## Files

- `python/functional_regression.py` — from-scratch FPC regression via
  SVD scores + OLS + reconstruction of `β(t)`. Demo: 200 curves from
  a 3-dim basis + noisy scalar target with true `β(t) = 2 sin(2π t)
  − t`. K = 3 FPCs recovers **scalar R² = 0.96** and
  **corr(β̂, β) = 1.000** (mean |diff| 0.031).
- `r/functional_regression.R` — `fda::fRegress`, `refund::pfr`,
  `fda.usc`, `FDboost` (R); `scikit-fda` (Python).

## Assumptions & caveats

- **K choice** — small K = under-fit β; large K = over-fit; use CV /
  scree.
- **Basis representation** — smoothing choice affects β̂ smoothness.
- **Penalty vs FPC route** — penalised methods work when curves are
  irregularly sampled.
- **Interpretability** — signed β(t) is easy to interpret; large-K
  fits can wiggle spuriously.
- **Function-on-function** and **function-on-scalar** are natural
  extensions.

## Related in this repo

- `functional-pca` — the score-computing sibling.
- `functional-anova`, `functional-clustering`, `curve-registration`,
  `functional-depth` — the FDA family (this batch).
- `pca-regression` (if present) — the multivariate cousin.
- `ridge-lasso-elasticnet`, `additive-quantile-regression` —
  regression flavours that stack with functional features.

## Run

```
python techniques/functional-regression/python/functional_regression.py
Rscript techniques/functional-regression/r/functional_regression.R
```

**Refs:** Ramsay, J.O. & Silverman, B.W. *Functional Data Analysis*, Springer, 2005 (Ch. 12-13); Goldsmith, J. et al. "Penalized functional regression." *Journal of Computational and Graphical Statistics*, 2011.

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
