# Conjugate Priors (Reference §14.1, §14.2, §14.3)

A prior is **conjugate** to a likelihood if the posterior belongs to the same family. Closed-form posteriors give the exact Bayesian answer without MCMC and are the go-to sanity check for any general-purpose sampler.

## Three canonical pairs

### Beta – Binomial

```
θ       ~ Beta(α, β)
y | θ   ~ Binomial(n, θ)
θ | y   ~ Beta(α + y, β + n − y)
posterior predictive:  Beta-Binomial(n_new, α_post, β_post)
```

### Gamma – Poisson

```
λ       ~ Gamma(α, rate = β)
yᵢ | λ  ~ Poisson(λ)
λ | y   ~ Gamma(α + Σyᵢ, β + n)
posterior predictive:  Negative-Binomial
```

### Normal (known σ²) – Normal

```
μ       ~ Normal(μ₀, τ₀²)
yᵢ | μ  ~ Normal(μ, σ²)
μ | y   ~ Normal(μ_n, τ_n²)         with 1/τ_n² = 1/τ₀² + n/σ²
                                      μ_n = τ_n² (μ₀/τ₀² + n ȳ/σ²)
posterior predictive (new obs):  Normal(μ_n, τ_n² + σ²)
```

## Default weakly-informative choices

- **Beta(1, 1)** — uniform prior on a proportion.
- **Gamma(0.5, 0.5)** — weakly informative on a rate; equivalent to a chi-square(1) prior.
- **Normal(0, 100²)** — diffuse prior on a location.

Gelman's advice: prefer weakly informative to fully flat; flat priors on log-hazards or logits often carry lots of prior mass at implausible values.

## Files

- `python/conjugate_priors.py` — closed-form Beta-Binomial, Gamma-Poisson, and Normal-Normal updates using `scipy.stats` for interval quantiles. Cross-check: posterior means match `scipy.stats.beta.mean` exactly.
- `r/conjugate_priors.R` — same three updates using base `pbeta` / `qbeta` / `pgamma` / `qgamma` / `qnorm`.

## When to use

- Quick sanity check on any MCMC / VI / EB sampler — the sampler must reproduce the analytic posterior.
- Small problems where the closed form is enough.
- Teaching Bayesian inference — the algebra shows how prior and data combine.

## Assumptions

- The likelihood family really is Binomial / Poisson / Normal — for over-dispersed counts, use Negative-Binomial or a hierarchical Gamma-Poisson.
- Independent observations conditional on the parameter.
- Known nuisance parameters (`σ²` in the Normal-Normal case). When `σ²` is unknown, use the Normal-Inverse-Gamma joint conjugate — implemented in `bayesian-linear-regression`.

## Run

```
python techniques/conjugate-priors/python/conjugate_priors.py
Rscript techniques/conjugate-priors/r/conjugate_priors.R
```

**Refs:** Gelman, A. et al. *Bayesian Data Analysis*, 3rd ed., CRC, 2013 (Ch 2–3); Hoff, P.D. *A First Course in Bayesian Statistical Methods*, Springer, 2009.

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
