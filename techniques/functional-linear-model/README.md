# Functional Linear Model (Reference §31.7)

Three flavours of FDA regression:

- **Scalar-on-function** — see `functional-regression`.
- **Function-on-scalar** — `Y_i(t) = α(t) + Σ β_p(t) x_ip + ε(t)`;
  coefficient **functions** regressed on scalar covariates.
- **Function-on-function** — `Y_i(t) = α(t) + ∫ X_i(s) β(s, t) ds +
  ε(t)`.

## Function-on-scalar via pointwise OLS + smoothing

```
1. For each t, fit OLS  β̂(t) = (XᵀX)⁻¹ Xᵀ Y[:, t].
2. Smooth β̂_p(·) via a P-spline penalty (see
   functional-basis-smoothing).
```

## When to use

- **Growth curves treated vs control** — β_1(t) is the treatment
  effect over the growth window.
- **Any function-valued outcome** with scalar covariates: hormone
  profiles, spectroscopy, sensor time-series.

## When NOT to use

- **Function-on-function is complicated** — needs 2-D basis for
  `β(s, t)`; specialised software (`refund::pffr`).
- **Sparse / irregular grids** — needs PACE-style methods.

## Files

- `python/functional_linear_model.py` — from-scratch pointwise OLS
  followed by P-spline smoothing of each coefficient trajectory.
  Demo: 2-group growth-curve setting where a shift function
  `β_1(t) = 0.6(t − 0.5) · 𝟙(t > 0.3)` only turns on later. Result:
  **coefficient MSE 0.008 → 0.001 with smoothing (83 % reduction)**;
  point estimates at `t = 0.8` = 0.183 vs truth 0.190; at `t = 0.1`
  = −0.023 vs truth 0.000.
- `r/functional_linear_model.R` — `fda::fRegress`, `refund::fosr`,
  `refund::pffr`, `FDboost` (R); `scikit-fda` (Python).

## Assumptions & caveats

- **Independent residuals across t** — if residuals correlate,
  pointwise-OLS SEs are wrong; use `refund::fosr` GLS variant.
- **Number of scalar covariates** — same for every `t`; extension to
  time-varying scalars via `varying-coefficient-model`.
- **Confidence bands** — pointwise vs simultaneous (Meyer 2015).
- **Function-on-function** — needs a 2-D basis (`refund::pffr`) or
  tensor B-spline (Ivanescu 2015).

## Related in this repo

- `functional-pca`, `functional-regression`, `functional-anova`,
  `functional-clustering`, `curve-registration`, `functional-depth`,
  `functional-basis-smoothing` — FDA family.
- `varying-coefficient-model` — closely related semiparametric
  alternative for scalar responses.
- `restricted-cubic-splines`, `additive-quantile-regression` — spline
  regression siblings.

## Run

```
python techniques/functional-linear-model/python/functional_linear_model.py
Rscript techniques/functional-linear-model/r/functional_linear_model.R
```

**Refs:** Ramsay, J.O. & Silverman, B.W. *Functional Data Analysis*, Springer, 2005 (Ch. 12-16); Reiss, P.T. et al. "Methods for scalar-on-function regression." *International Statistical Review*, 2017; Goldsmith, J. et al. "Penalized functional regression." *JCGS*, 2011.

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
