# Varying-Coefficient Model (Reference §33.9)

Each regression coefficient is a **smooth function of an
effect-modifier `U`**. Hastie & Tibshirani (1993) named the family.

## Model

```
Y  =  β₀(U) + β₁(U) X₁ + … + β_p(U) X_p + ε
```

- **`β_j(u) ≡ constant`** ⇒ ordinary linear regression.
- **`U = t` (time)** ⇒ time-varying coefficient models.
- **`U = spatial coord`** ⇒ geographically-weighted regression.
- **`U = age`, `U = dose`** ⇒ effect modification along a smooth axis.

## Local WLS estimation

At each query point `u₀`, weight observations by kernel `K_h(u_i − u₀)`
and solve weighted normal equations:

```
β̂(u₀)  =  argmin_β  Σ_i K_h(u_i − u₀) · ( y_i − x_iᵀ β )²
```

## When to use

- **Effect modifiers** you want to visualise (dose × response, age ×
  treatment).
- **Time-varying regression** — a lightweight state-space alternative.
- **Geographically-weighted regression** — the spatial-statistics
  cousin.

## When NOT to use

- **Multiple effect modifiers** — becomes multivariate smoothing;
  interpretability drops.
- **Interactions between the modifier and multiple X's** — VCM handles
  it, but the coefficient surface can be noisy.
- **Very high-dim X** — bandwidth tuning becomes fragile.

## Files

- `python/varying_coefficient_model.py` — from-scratch local Gaussian-
  kernel WLS at 9 query values of `u`. True `β₀(u) = sin(u)`,
  `β₁(u) = u/2`. Result: both coefficient functions are recovered
  smoothly across the queried grid.
- `r/varying_coefficient_model.R` — `mgcv::gam(y ~ s(u, by=x))`,
  `svcm`, `BayesX`, `np` (R); `pyGAM` (Python).

## Assumptions & caveats

- **Bandwidth `h`** — small ⇒ noisy; large ⇒ over-smoothed. Cross-
  validation or plug-in bandwidth is standard.
- **Boundary bias** — local constant fits under-shoot at the edges;
  local linear (loess-style) reduces the bias.
- **Standard errors** — sandwich SEs from local WLS; or bootstrap over `i`.
- **Regularisation** — for wiggly coefficient functions, use P-spline
  penalties (`mgcv`) rather than raw local WLS.
- **Model comparison** — nested test against constant coefficients via
  approximate F-test on residual variance drop.

## Related in this repo

- `single-index-model` — a projection cousin.
- `additive-quantile-regression`, `distributional-regression` —
  flexible non-linear regression families.
- `local-regression-loess`, `kernel-density-estimation` — the smoothers
  in the same family.
- `spatial-statistics-*` (if present) — spatially-varying-coefficient
  regressions are the geospatial special case.

## Run

```
python techniques/varying-coefficient-model/python/varying_coefficient_model.py
Rscript techniques/varying-coefficient-model/r/varying_coefficient_model.R
```

**Refs:** Hastie, T. & Tibshirani, R. "Varying-coefficient models." *Journal of the Royal Statistical Society, Series B*, 1993; Fan, J. & Zhang, W. "Statistical methods with varying coefficient models." *Statistics and Its Interface*, 2008.

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
