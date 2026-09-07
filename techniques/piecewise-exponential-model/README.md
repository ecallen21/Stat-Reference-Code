# Piecewise Exponential Model (Reference §11.26)

Friedman (1982); Holford (1980). Approximates any continuous hazard
by a step function: partition follow-up time into `K` intervals with
constant hazard within each:

    h(t | x) = λ_k · exp(x' β)     for t ∈ [a_{k−1}, a_k)

Fit via **Poisson GLM on split data**: expand each subject into one
row per interval they enter, with offset `log(exposure time)` and
outcome = event indicator in that interval.

## Files

- `python/piecewise_exponential_model.py` — person-time splitter +
  Poisson MLE from scratch. Demo (n=800, true λ=(0.20, 0.50, 1.00)
  on cuts (0, 1, 3, ∞), HR=2): recovers λ̂=(0.201, 0.502, 0.81) and
  ĤR=2.09.
- `r/piecewise_exponential_model.R` — `survival::pyears + glm(family=
  poisson)`, `eha::phreg`, `pch`, `rstpm2`, `flexsurv` (R);
  `lifelines.PiecewiseExponentialRegressionFitter` (Python).

## When to use

- **Any survival analysis where hazard shape matters** — non-monotone
  hazards are the common case in clinical follow-up.
- **Time-varying covariates** — easy to include as extra columns on
  the split dataset.
- **Bayesian survival** — piecewise-exponential likelihoods are
  tractable in Stan / JAGS.
- **Excess-hazard / relative-survival** — additive-hazard models
  build cleanly on the piecewise framework.

## When NOT to use

- **Very few events** — many intervals × few events → unstable
  `λ_k` estimates; use fewer cuts or a smoothing penalty.
- **You need PH interpretation only** — Cox is more parsimonious;
  piecewise-exp adds only when you care about baseline hazard.

## Assumptions & caveats

- **Cutpoints** — usually quantiles of event times, or clinically
  motivated. Sensitivity check with different partitions.
- **Constant hazard within intervals** — bias-vs-variance
  trade-off; more cuts reduce bias, raise variance.
- **Time-varying covariates** — must be measured at each cut (or
  approximated).
- **Poisson GLM equivalence** — the split-and-Poisson trick gives
  MLE identical to direct piecewise-exponential MLE (Aitkin & Clayton
  1980).

## Related in this repo

- `cox-ph`, `cox-time-varying`, `accelerated-failure-time` —
  competing survival models.
- `parametric-survival` — the umbrella; piecewise-exponential is one
  parametric form.
- `additive-aalen` — additive hazard cousin.
- `poisson-regression` — the fitting engine.

## Run

```
python techniques/piecewise-exponential-model/python/piecewise_exponential_model.py
Rscript techniques/piecewise-exponential-model/r/piecewise_exponential_model.R
```

**Refs:** Friedman, M. "Piecewise exponential models for survival data with covariates." *Annals of Statistics*, 10(1): 101-113, 1982; Holford, T.R. "The analysis of rates and of survivorship using log-linear models." *Biometrics*, 36(2): 299-305, 1980.

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
