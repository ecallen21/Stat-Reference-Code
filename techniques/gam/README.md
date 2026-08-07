# Generalized Additive Models (Reference §5.14)

```
g(E[y_i]) = β_0 + Σ_j f_j(x_ij) + ε_i
```

Each `f_j` is a **smooth function** learned from the data. Combines the interpretability of linear regression with the flexibility of nonparametric smoothing. `g` is a link (identity for Gaussian, logit for binomial, log for Poisson).

## Basis + penalty formulation

Each smooth is expanded on a fixed basis (cubic regression splines, B-splines, thin-plate splines):

```
f_j(x) = Σ_k b_{jk}(x) β_{jk}
```

and shrunk by an integrated-second-derivative penalty:

```
λ_j ∫ f_j''(x)² dx  =  β_jᵀ S_j β_j
```

The smoothing parameter `λ_j` is chosen by **GCV** or **REML** (Wood 2017). Large `λ_j` → linear; small `λ_j` → wiggly. Effective degrees of freedom (edf) = trace of the hat matrix.

## Files

- `python/gam.py` — cubic-spline basis + integrated-quadratic penalty + GCV grid search. Demo (n = 300, two nonlinear terms `sin(1.5x₁) + 0.3(x₂ − 5)² − 3`): in-sample RMSE 0.45 (noise sd 0.5); correlations between fitted and true smooths = 0.998 / 0.999.
- `r/gam.R` — `mgcv::gam(y ~ s(x1) + s(x2), method = "REML")` — Simon Wood's canonical implementation with automatic smoothness selection, GLM support, tensor-product interactions, random effects.

## When to use

- **Nonlinear** covariate effects where a specific parametric form isn't credible.
- **Dose-response curves** with unknown shape.
- **Epidemiology**: adjust for age, BMI, temperature, seasonality via smooth terms.
- **Sensor / physical models** with smooth confounders.

## Related methods

- **Splines** with fixed knot count and no penalty (`splines-regression`).
- **LOESS** — local polynomial smoother; no explicit basis.
- **Tensor products** for interactions between smooth terms: `s(x1, x2)`.
- **GAMMs** — GAM with random effects (`mgcv::gamm` or `gamm4::gamm4`).

## Assumptions & caveats

- **Basis dimension**: `k` should be large enough that the penalty (not `k`) controls smoothness; check with `mgcv::gam.check`.
- **Identifiability**: each smooth is centered to sum to zero (constant is absorbed into the intercept) — makes term plots comparable.
- **Extrapolation** beyond the data range is unreliable; smooth curves can shoot off to plus/minus infinity.
- **Concurvity**: two smooths of correlated covariates can trade off; check with `mgcv::concurvity`.

## Run

```
python techniques/gam/python/gam.py
Rscript techniques/gam/r/gam.R
```

**Refs:** Hastie, T. & Tibshirani, R. *Generalized Additive Models*, Chapman & Hall, 1990; Wood, S.N. *Generalized Additive Models: An Introduction with R*, 2nd ed., CRC, 2017.

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
