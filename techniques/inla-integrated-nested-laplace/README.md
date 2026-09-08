# INLA -- Integrated Nested Laplace Approximation (Reference §25.13)

Rue, Martino & Chopin (2009, JRSS-B). Fast deterministic alternative
to MCMC for models with a **Gaussian latent field**:

    y | x, θ ~ likelihood(x, θ)      observations
    x | θ    ~ N(0, Q(θ)⁻¹)           Gaussian Markov random field
    θ         ~ π(θ)                   hyperparameters

## Steps

1. Laplace-approximate `p(θ | y)` via posterior mode + Hessian.
2. For each `θ_k` on a grid, Laplace-approximate `p(x_i | θ_k, y)`.
3. Integrate: `p(x_i | y) ≈ Σ w_k · p(x_i | θ_k, y)`.

## Files

- `python/inla_integrated_nested_laplace.py` — mini-INLA:
  Gaussian latent + likelihood + grid over precision from scratch.
  Demo (n=50, x ∼ N(0, 1), y = x + N(0, 1)): INLA weights peak
  around the true precision; posterior mean MSE 0.524 vs oracle
  Bayes with true hyperparameter 0.514.
- `r/inla_integrated_nested_laplace.R` — `INLA` (r-inla.org),
  `inlabru`, `brinla` (R); no pip-installable port; describes
  pymc / numpyro closest analogues (Python).

## When to use

- **Spatial epidemiology** — BYM / BYM2 disease mapping.
- **Smoothing splines / penalised regression** with Gaussian
  priors.
- **Geostatistics with SPDE** — INLA-SPDE for continuous Gaussian
  random fields.
- **Hierarchical GLMs** — much faster than MCMC when the latent
  field is Gaussian.
- **Reproducible marginal posteriors** — deterministic; no
  autocorrelation to diagnose.

## When NOT to use

- **Non-Gaussian latent** (t-priors, discrete random effects) —
  not the target class.
- **Very high-dim hyperparameters (> ~7)** — grid integration
  scales poorly; use HMC.
- **Small data with strong priors** — the Laplace step's normal
  approximation degrades.

## Assumptions & caveats

- **GMRF latent** — Q(θ) must be sparse; SPDE conversions supply
  it for continuous fields.
- **Skewed / heavy-tailed marginals** — use simplified Laplace
  (SLA) or full Laplace with skew correction.
- **Model selection** — DIC, WAIC, LOO-CV via `INLA` output;
  usually cheaper than MCMC.
- **Not a hammer** — for genuinely non-Gaussian latent, MCMC (HMC,
  ESS) is still preferable.

## Related in this repo

- `bayesian-hierarchical-models`, `laplace-approximation`,
  `variational-inference`, `bridge-sampling-evidence`,
  `nested-sampling`, `elliptical-slice-sampling`,
  `hmc-nuts` — Bayesian computation cousins.
- `conditional-autoregressive-car`, `spatial-glm`,
  `poisson-gamma-empirical-bayes` — spatial GLMs that INLA fits
  natively.

## Run

```
python techniques/inla-integrated-nested-laplace/python/inla_integrated_nested_laplace.py
Rscript techniques/inla-integrated-nested-laplace/r/inla_integrated_nested_laplace.R
```

**Refs:** Rue, H., Martino, S. & Chopin, N. "Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations." *JRSS-B*, 71(2): 319-392, 2009; Rue, H. et al. "Bayesian computing with INLA: a review." *Annual Review of Statistics*, 4: 395-421, 2017.

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
