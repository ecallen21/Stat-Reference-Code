# Laplace Approximation (Reference §14.29)

Approximate an intractable posterior `p(θ | y)` by a **Gaussian centered at the MAP** with covariance equal to the inverse of the negative Hessian of the log posterior at the MAP:

```
log p(θ | y) ≈ log p(θ_MAP | y) − ½ (θ − θ_MAP)ᵀ H (θ − θ_MAP)
p(θ | y)    ≈ N(θ_MAP,  H⁻¹)
```

## Marginal likelihood (evidence)

```
log p(y) ≈ log p(y | θ_MAP) + log p(θ_MAP) + (d / 2) log(2π) − ½ log|H|
```

Useful for Bayes factors and BMA when integrating the posterior directly is expensive.

## INLA (Rue-Martino-Chopin 2009)

**Integrated Nested Laplace Approximation** applies nested Laplace approximations to **latent-Gaussian models** — a very broad class covering GAMs, Cox models with baseline splines, spatial models, hierarchical GLMs. Fast Bayes without MCMC, matched to Stan/PyMC on many benchmarks.

- **R-INLA** (`INLA::inla`) is the canonical implementation.
- Python bridges exist (`inlabru`, `sdmTMB`) but R remains the primary interface.

## Files

- `python/laplace_approximation.py` — MAP via BFGS + numerical Hessian + log-evidence approximation. Demos:
  - Beta-Binomial (a₀ = b₀ = 2, y = 8/12): Laplace MAP = 0.643 matches analytic Beta mode 0.643 exactly; SD 0.128 close to analytic 0.117.
  - Bayesian logistic (n = 200, β = (0.5, 1.5)): recovers β̂ = (0.25, 1.62) with SD (0.17, 0.24).
- `r/laplace_approximation.R` — base R `optim(..., hessian = TRUE)` + `solve(hess)` for the Laplace-Gaussian approximation. Notes on `INLA` for production use.

## When to use

- **Fast Bayesian inference** when MCMC is too expensive but the posterior is roughly Gaussian.
- **Marginal-likelihood approximation** for Bayes factors / BMA over many candidate models.
- **INLA** for latent-Gaussian models: hierarchical GLMs, spatial models, spline-based GAMs — the standard "fast Bayesian" alternative to Stan.

## When NOT to use

- **Multimodal posteriors** — Laplace picks whichever mode `optim` lands in and misses the others.
- **Skewed / boundary posteriors** — the Gaussian approximation is symmetric, so it misfits skew.
- **Small sample sizes** — the "posterior is asymptotically Gaussian" heuristic requires `n` large enough for the log-likelihood to dominate the prior.

## Improvements beyond plain Laplace

- **Variational Bayes** — replaces the Gaussian by any tractable family; can capture skew (see `variational-inference`).
- **Skew-normal Laplace / Cornish-Fisher expansion** — corrects for skew at the MAP.
- **Importance sampling with Laplace as proposal** — uses Laplace as a starting distribution, reweights toward the true posterior.
- **INLA nested-Laplace** — applies Laplace to conditional posteriors, integrates numerically over the hyperparameter marginals.

## Run

```
python techniques/laplace-approximation/python/laplace_approximation.py
Rscript techniques/laplace-approximation/r/laplace_approximation.R
```

**Refs:** Tierney, L. & Kadane, J.B. "Accurate approximations for posterior moments and marginal densities." *JASA* 81(393), 82–86, 1986; Rue, H., Martino, S. & Chopin, N. "Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations." *J. R. Stat. Soc. B* 71(2), 319–392, 2009.

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
