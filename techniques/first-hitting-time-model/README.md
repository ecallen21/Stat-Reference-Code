# First-Hitting-Time Model (Reference §11.28)

Whitmore (1986); Aalen, Borgan & Gjessing (2008). Models the failure
time as the **first time a latent process** `X(t)` hits an absorbing
boundary.

For **Brownian motion with drift** `−μ` (μ > 0 wear), starting at
`X(0) = y₀ > 0` and failing at `X = 0`, the failure time follows an
**inverse-Gaussian**:

    T ~ IG(y₀/μ, y₀²/σ²)
    E[T] = y₀/μ,   Var(T) = y₀ σ² / μ³

Density:

    f(t) = y₀ / √(2π σ² t³) · exp(−(y₀ − μ t)² / (2 σ² t))

Cure fraction: `P(T = ∞) = 1 − exp(−2 y₀ μ / σ²)` allowed if drift
can be negative.

## Files

- `python/first_hitting_time_model.py` — closed-form MoM MLE for
  uncensored data + Nelder-Mead MLE for right-censored inverse-
  Gaussian using scipy from scratch. Demo (n=1000, true μ=3.0,
  λ=10): uncensored MLE (3.065, 9.729); censored (42%) MLE
  (3.041, 9.822).
- `r/first_hitting_time_model.R` — `threg`, `flexsurv::flexsurvreg`,
  `SMPracticals` (R); `scipy.stats.invgauss`, lifelines
  InverseGaussianAFTFitter (Python).

## When to use

- **Latent-process interpretation** — wear-out, cognitive decline,
  disease progression modeled as damage accumulation.
- **Unimodal hazard shapes** — hazard rises then falls; Weibull
  can't do this monotonically.
- **Cure-fraction settings** — drift-permits-escape gives natural
  long-term survivors.
- **Reliability / manufacturing** — first-passage-time is standard.

## When NOT to use

- **Monotone hazards** — Weibull / Gompertz are simpler.
- **Discrete or event-count data** — need continuous-time process.
- **You want interpretability of covariates on hazard** — Cox with
  splines or GAM is more transparent.

## Assumptions & caveats

- **Brownian-motion assumption** on latent process — extensions to
  jump / Levy processes exist but complicate fitting.
- **Boundary interpretation** — the "damage threshold" is a modelling
  choice; scale-identifiability requires anchoring `y₀ = 1`.
- **Right-censoring** handled via CDF in the likelihood; interval
  censoring needs the between-times CDF differences.
- **Regression links** — Whitmore parameterises `log μ = X β` for
  hazard-affecting covariates.

## Related in this repo

- `parametric-survival`, `accelerated-failure-time`, `cox-ph` —
  survival competitors.
- `cure-models` — cousin allowing long-term survivors.
- `state-space-kalman`, `state-space-models` — latent-process
  siblings.
- `stochastic-volatility` — another SDE / diffusion inference
  problem.

## Run

```
python techniques/first-hitting-time-model/python/first_hitting_time_model.py
Rscript techniques/first-hitting-time-model/r/first_hitting_time_model.R
```

**Refs:** Whitmore, G.A. "First passage time models for duration data: regression structures and competing risks." *The Statistician*, 35(2): 207-219, 1986; Aalen, O.O., Borgan, O. & Gjessing, H.K. *Survival and Event History Analysis*, Springer, 2008.

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
