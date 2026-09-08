# Beta-Binomial Hierarchical Model (Reference §47.132)

Efron & Morris (1975); Gelman-Carlin-Stern-Rubin (2013 ch 5). For
group i with x_i successes out of n_i:

    pᵢ | α, β ~ Beta(α, β)
    xᵢ | pᵢ, nᵢ ~ Binomial(nᵢ, pᵢ).

Marginal is BetaBinomial(n, α, β). Empirical- or full-Bayes
posterior mean per group SHRINKS noisy MLE `x/n` toward the pooled
mean; shrinkage strength = α + β.

## Files

- `python/beta_binomial_hierarchical.py` — empirical-Bayes fit
  by BetaBinomial marginal-likelihood MLE (`scipy.special.betaln`).
  Demo on the Efron-Morris (1975) baseball data (18 players, first
  45 at-bats):
  - EB hyperparams α = 15.3, β = 40.2 (α+β ≈ 45)
  - **Held-out MSE: MLE 0.0036 vs EB shrunk 0.0010 (71 % better)**.
- `r/beta_binomial_hierarchical.R` — `VGAM::vglm(family=betabinomial)`,
  `brms`, `rstanarm` (R); `pymc`, `stan` (Python).

## When to use

- **Many groups with few per-group observations** — small-area
  estimation, sports batting averages, click-through-rates.
- **Sparse counts** — shrinkage denoises.
- **Bayes-factor / hierarchical Bayesian analyses** on proportions.
- **A/B testing at scale** across many variants.

## When NOT to use

- **Single group** — no hierarchy; use Bayesian binomial.
- **Underdispersion** — beta-binomial only handles OVER-dispersion;
  consider Conway-Maxwell binomial.
- **Extreme n_i imbalance** — hierarchy can dominate small groups
  entirely.
- **When covariates matter** — extend to hierarchical logistic
  regression.

## Assumptions & caveats

- **Exchangeability** of groups under the prior — verify by
  checking residual structure.
- **Empirical vs full Bayes** — EB understates uncertainty; full
  Bayes gives proper credible intervals.
- **Prior on (α, β)** — flat on log(α+β), log(α/β) is a common
  weakly-informative choice.
- **Model checking**: posterior-predictive p-values, DIC / WAIC.

## Related in this repo

- `bayesian-hierarchical-models`, `bayesian-glms`,
  `bayesian-linear-regression`, `bayesian-optimization` —
  hierarchical Bayes cousins.
- `james-stein-shrinkage`, `mrp-poststratification`,
  `fay-herriot-small-area`, `poisson-gamma-empirical-bayes`,
  `buhlmann-credibility`, `buhlmann-straub-credibility` —
  small-area / shrinkage neighbours.
- `wilson-score-interval-proportion`, `firth-logistic`,
  `binomial-test`, `rates-proportions` — proportion inference
  cousins.

## Run

```
python techniques/beta-binomial-hierarchical/python/beta_binomial_hierarchical.py
Rscript techniques/beta-binomial-hierarchical/r/beta_binomial_hierarchical.R
```

**Refs:** Efron, B. & Morris, C. "Data analysis using Stein's estimator and its generalizations." *JASA* 70(350): 311-319, 1975; Gelman, A. et al. *Bayesian Data Analysis*, 3rd ed., CRC Press, 2013, ch 5.

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
