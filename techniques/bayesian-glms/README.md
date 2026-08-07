# Bayesian GLMs (Reference §14.12, §14.13)

Generalized linear models with a prior on the coefficient vector:

```
y_i | X, β  ~ ExponentialFamily(link⁻¹(x_iᵀ β))
β           ~ Normal(0, s² I)
```

No conjugate prior exists in general, so we sample from the posterior. Two default samplers used here:

- **MH with Laplace-approximation proposal** — MAP + inverse-Hessian gives a well-scaled multivariate proposal; simple and fast for moderate `p`.
- **HMC / NUTS** in production (`rstanarm`, `brms`, `PyMC`, `NumPyro`) — better for high-dimensional or strongly correlated posteriors.

## Weakly informative default priors (Gelman 2008)

```
β_intercept ~ Normal(0, 10²)
β_slope     ~ Normal(0, 2.5²)      (on STANDARDIZED covariates)
```

Standardize continuous predictors to make the `2.5²` scale meaningful. For binary predictors, Gelman suggests `Cauchy(0, 2.5)` — heavier tails absorb rare extreme signals.

## Files

- `python/bayesian_glms.py` — MH sampler with Laplace-approx proposal for logistic and Poisson GLMs. Demo (n = 300): Poisson posterior means (0.441, 0.655, −0.438) match `statsmodels.GLM` MLE (0.442, 0.655, −0.437) to 3 decimals; logistic recovers all three slope signs and magnitudes.
- `r/bayesian_glms.R` — same MH loop in base R. Production: `rstanarm::stan_glm(family=binomial)` / `arm::bayesglm`.

## When to use

- Small-to-moderate GLMs with informative priors (rare-event logistic, sparse-data Poisson).
- When you want honest **posterior predictive intervals**, not just standard errors.
- Regularization framed probabilistically — `Normal(0, s²)` prior on log-odds ≡ ridge on logistic.
- Hierarchical GLMs (multilevel logistic / Poisson) — combine with `bayesian-hierarchical-models`.

## Diagnostics

- Acceptance rate 0.2–0.4 is fine for MH; NUTS should reach 0.6–0.9.
- Trace plots + Gelman-Rubin R-hat + ESS.
- **Posterior predictive checks** (see `posterior-predictive-checks`).

## Run

```
python techniques/bayesian-glms/python/bayesian_glms.py
Rscript techniques/bayesian-glms/r/bayesian_glms.R
```

**Refs:** Gelman, A., Jakulin, A., Pittau, M.G. & Su, Y.-S. "A weakly informative default prior distribution for logistic and other regression models." *Ann. Appl. Stat.* 2(4), 1360–1383, 2008; Gelman, A. et al. *Bayesian Data Analysis*, 3rd ed., CRC, 2013 (Ch 16).

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
