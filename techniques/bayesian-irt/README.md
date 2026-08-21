# Bayesian IRT (Reference §22.x extra)

Add priors to the classical IRT likelihood:

- **Ability**: `θ_i ~ N(0, 1)` — resolves scale identifiability *and* provides shrinkage on short tests.
- **Discrimination**: `log(a_j) ~ N(0, τ_a²)` — positivity + moderate shrinkage.
- **Difficulty**: `b_j ~ N(0, τ_b²)` — moderate shrinkage toward the test mean.

## Why Bayesian IRT

- **Small samples** — MLE for 2PL / 3PL is unstable; priors regularise `a` and `c` (guessing).
- **Uncertainty quantification** — posterior credible intervals for `θ, a, b`, item-fit statistics, PPP p-values.
- **Missing / sparse data** — trivial to handle; MCMC integrates over latent structure.
- **Hierarchical extensions** — grouping items by content, persons by school; measurement invariance testing.
- **Computerised adaptive testing** — item-selection uses the posterior variance of `θ` after each response.

## Fitting

- **Full MCMC (Stan)** — HMC/NUTS on the joint posterior. Slow but exact. `edstan`, `brms`, `pymc`, `numpyro`.
- **Polya-Gamma Gibbs (Polson-Scott-Windle 2013)** — closed-form conditional draws using the PG(1, ψ) augmentation. Requires a PG sampler (`pypolyagamma`).
- **MAP / coordinate ascent** — this module. Fast, gives the posterior mode without draws; suitable for point summaries and warm-starts, not for credible intervals.
- **Variational Bayes** — mean-field approximate posterior; faster than MCMC, underestimates variance.

## Files

- `python/bayesian_irt.py` — 2PL MAP via coordinate-ascent Newton with priors `log(a) ~ N(0, 0.25)`, `θ ~ N(0, 1)`, `b ~ N(0, 4)`; explicit identification by rescaling `θ` to sd 1 (and compensating `a`, `b`). Demo (n=400, J=15): cor(a_hat, a) = 0.87, cor(b_hat, b) = 0.98, cor(θ_hat, θ) = 0.84; item parameters recovered to their true scale.
- `r/bayesian_irt.R` — `brms::brm`, `edstan::stan_rasch / stan_2pl`, `rstan`, `mirt::mirt(method='Bayesian')`.

## Assumptions & caveats

- **MAP ≠ posterior mean** — for skewed marginals (e.g. `log(a)`) the mode and mean differ; report both when you can, or run MCMC for the mean and intervals.
- **Prior sensitivity** — `τ_a` is the single most influential prior. Half-Cauchy or exponential on `τ_a` (a hyperprior) lets the data speak.
- **Identifiability** requires anchoring — fixing `θ ~ N(0, 1)` or anchoring one item's `a = 1`, `b = 0`. Without it MCMC will produce a sign-flipped multi-modal posterior.
- **Guessing (3PL) is often weakly identified** — put an informative Beta(5, 17) or Beta(20, 80) prior on `c_j`.
- **Model checking** — posterior predictive checks (item p-values, biserial, WAIC / LOO for model choice) — see `posterior-predictive-checks` and `bayesian-model-comparison`.

## Related in this repo

- `rasch-model`, `two-three-pl-irt`, `graded-response-model`, `partial-credit-model` — frequentist MML fits.
- `mcmc-metropolis-hastings`, `gibbs-sampler`, `hamiltonian-mc` — general Bayesian samplers.
- `variational-inference`, `laplace-approximation` — approximate posteriors.
- `bayesian-model-comparison`, `posterior-predictive-checks` — model choice and fit.

## Run

```
python techniques/bayesian-irt/python/bayesian_irt.py
Rscript techniques/bayesian-irt/r/bayesian_irt.R
```

**Refs:** Fox, J.-P. *Bayesian Item Response Modeling*, Springer, 2010; Polson, N.G., Scott, J.G. & Windle, J. "Bayesian inference for logistic models using Pólya-Gamma latent variables." *JASA* 108(504), 1339–1349, 2013; Bürkner, P.-C. "Bayesian Item Response Modeling in R with brms and Stan." *J. Stat. Softw.* 100(5), 2021.

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
