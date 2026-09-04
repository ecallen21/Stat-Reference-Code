# Measurement Error / Errors-in-Variables Models (Reference §38.7)

Carroll, Ruppert, Stefanski & Crainiceanu (2006). When predictors are
measured with error, ordinary regression is biased. Two structures
matter:

- **Classical** — `W = X + U` (surrogate scatters around true).
  Attenuates the naive slope.
- **Berkson** — `X = W + U` (true scatters around assigned value).
  Preserves slope; inflates residual variance.

## Attenuation

Under classical error and normal `X`,

```
β̂_naive = β · λ,    λ = σ_X² / (σ_X² + σ_U²)  <  1
```

## Corrections

- **Regression calibration** — replace `W` by `E[X | W]` (linear in
  `W` under joint normality) and refit; divides the naive slope by
  `λ̂`.
- **SIMEX** (Cook-Stefanski 1994) — add extra noise at multiple
  levels `ζ`, refit each, extrapolate `ζ → −1`.
- **Bayesian** — put a prior on the latent `X`; useful for
  nonlinear models.

## When to use

- **Self-reported exposure** (diet, activity, income) in
  epidemiology.
- **Instrument noise** in physical measurements.
- **Proxy variables** in econometrics.

## When NOT to use

- **Berkson error** — usually no correction needed for the slope.
- **Error in the outcome** — inflates SE but does not bias slopes
  under standard regression assumptions.
- **No estimate of `σ_U²`** — corrections require a validation
  sub-study, repeated measures, or a known instrument.

## Files

- `python/measurement_error_models.py` — regression calibration +
  SIMEX. Demo (n=500, β₁=1.5, σ_U=0.7): naive OLS β̂ = **1.06**
  (attenuated by λ̂ = 0.67); regression-calibration corrects to
  **1.59**; SIMEX to **1.36**.
- `r/measurement_error_models.R` — `simex`, `mecor`, `merror`,
  `eivtools` (R); `scipy.odr` + custom (Python).

## Assumptions & caveats

- **Additive independent error** — non-standard error structures
  (differential, correlated) need alternatives.
- **Correct `σ_U²`** — a wrong error variance produces wrong
  correction; validation studies are the gold standard.
- **Regression calibration is exact only for linear regression**
  with normal errors; generalises approximately (RC + Taylor
  expansion) for nonlinear models.
- **SIMEX** is more general (nonlinear models, non-normal `X`) but
  computationally heavier and depends on the extrapolant class.

## Related in this repo

- `orthogonal-distance-regression` (via `scipy.odr`) — errors in
  both `X` and `Y`.
- `bayesian-glms` — natural home for Bayesian error models.
- `instrumental-variables` — different remedy under a different
  assumption structure.

## Run

```
python techniques/measurement-error-models/python/measurement_error_models.py
Rscript techniques/measurement-error-models/r/measurement_error_models.R
```

**Refs:** Carroll, R.J., Ruppert, D., Stefanski, L.A., & Crainiceanu, C.M. *Measurement Error in Nonlinear Models*, 2nd ed., Chapman & Hall/CRC, 2006; Fuller, W.A. *Measurement Error Models*, Wiley, 1987; Cook, J.R. & Stefanski, L.A. "Simulation-extrapolation estimation in parametric measurement error models." *JASA*, 1994.

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
