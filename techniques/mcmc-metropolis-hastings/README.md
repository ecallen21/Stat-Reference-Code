# Metropolis-Hastings MCMC (Reference §14.6)

General-purpose sampler for arbitrary unnormalized log-densities `log p(θ)`. Constructs a Markov chain whose stationary distribution is the target — the workhorse of Bayesian computation before HMC/NUTS became widespread.

## Random-walk Metropolis

```
Propose  θ' = θ + ε           ε ~ N(0, Σ_prop)
Accept   with prob min{1, p(θ') / p(θ)}
```

## Haario adaptive proposal (Haario et al. 2001)

Every few hundred iterations, update `Σ_prop` to the sample covariance of the chain so far, scaled by the optimal `(2.38)² / d` (Roberts, Gelman & Gilks 1997). Adaptation must be turned off after burn-in (or diminish over time) to preserve stationarity.

## Diagnostics

- **Acceptance rate** — target ≈ 0.44 in 1-D, ≈ 0.234 in high dimension.
- **Trace plots** — visual check for mixing.
- **Effective sample size** (Geyer's initial monotone sum): `ESS = N / (1 + 2 Σ ρ_k)`.
- **Gelman-Rubin R-hat** across multiple chains: near 1.0 = converged.

## Files

- `python/mcmc_metropolis_hastings.py` — random-walk MH with Haario adaptation, ESS, and R-hat. Demos: recovers N(3, 1.5²) with ESS ≈ 1100 out of 4500 samples, R-hat ≈ 1.000 across 3 chains; also recovers a 2-D correlated Gaussian (empirical cov matches truth).
- `r/mcmc_metropolis_hastings.R` — same MH with adaptive covariance; ESS via `coda::effectiveSize`.

## When to use

- Any target where the log-density is computable but the normalizing constant isn't.
- Small problems, custom likelihoods, discrete latent variables — HMC needs gradients.
- Pedagogy — the accept/reject step makes the Bayesian invariance concrete.

## When to prefer HMC / NUTS

- Continuous, moderate-to-high-dimensional targets.
- Strong posterior correlations where random-walk MH mixes poorly.

## Assumptions & caveats

- **Detailed balance** — the symmetric random-walk proposal makes the Hastings ratio the plain likelihood ratio; for asymmetric proposals include the proposal ratio.
- **Discard burn-in** — the initial transient is not from the stationary distribution.
- **Stop adapting** — either fix `Σ_prop` after warm-up or use diminishing adaptation.

## Run

```
python techniques/mcmc-metropolis-hastings/python/mcmc_metropolis_hastings.py
Rscript techniques/mcmc-metropolis-hastings/r/mcmc_metropolis_hastings.R
```

**Refs:** Metropolis, N. et al. "Equation of state calculations by fast computing machines." *J. Chem. Phys.* 21(6), 1087–1092, 1953; Hastings, W.K. "Monte Carlo sampling methods using Markov chains and their applications." *Biometrika* 57(1), 97–109, 1970; Haario, H., Saksman, E. & Tamminen, J. "An adaptive Metropolis algorithm." *Bernoulli* 7(2), 223–242, 2001; Roberts, G.O., Gelman, A. & Gilks, W.R. "Weak convergence and optimal scaling of random walk Metropolis algorithms." *Ann. Appl. Probab.* 7(1), 110–120, 1997.

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
