# SMC Samplers (Reference §47.372)

Del Moral, Doucet & Jasra (2006). Turn MCMC into a particle
system by MOVING through a sequence of intermediate
distributions:

```
π₀ (easy)  →  π₁  →  …  →  π_T (target)
```

Popular annealing / tempering: `π_t ∝ prior · likelihood^{λ_t}`
with `0 = λ_0 < λ_1 < … < λ_T = 1`. Each step:

1. **Reweight** by `π_{t+1} / π_t`
2. **Resample** if effective-sample-size drops below N/2
3. **Move** particles via MCMC targeting `π_{t+1}` for
   rejuvenation.

Also yields an UNBIASED estimate of the marginal likelihood
`p(data) = Π E[w_t]`.

## Files

- `python/smc_samplers_del_moral.py` — Conjugate Normal-
  Normal posterior with 10 observations. Analytical
  posterior `N(2.861, 0.315²)` recovered by SMC with
  1 000 particles and 20-step temperature schedule:
  `mean = 2.836, sd = 0.313`.
- `r/smc_samplers_del_moral.R` — `SMC::sequential_MC`,
  `particles`, `nimbleSMC` (R); `particles.Chopin`,
  `blackjax.tempered_smc`, from-scratch (Python).

## When to use

- **Multi-modal posteriors** — tempering visits high-
  temperature modes before cooling to the target.
- **Marginal likelihood / model evidence** — unbiased log-Z
  estimate falls out for free.
- **Sequential / streaming inference** — natural fit when
  data arrives in batches.
- **Rare-event probability estimation** — using splitting or
  cross-entropy variants.

## When NOT to use

- **Very high-dim continuous smooth targets** — HMC / NUTS is
  more efficient per-gradient.
- **When only cheap point estimates are needed** —
  variational inference is faster.
- **When MCMC mixes well on the full target** — plain MCMC is
  simpler.

## Assumptions & caveats

- **Temperature schedule** — geometric or adaptive (target
  ESS every step) are standard; too-coarse ⇒ weight collapse.
- **Rejuvenation kernel** — a few MH / HMC sweeps at each
  temperature; without this, particles collapse to a few
  ancestors.
- **Resampling scheme** — multinomial, stratified,
  systematic; systematic has lowest variance.
- **ESS threshold** — resample only when `ESS < N/2` to
  avoid unnecessary Monte-Carlo variance.
- **Marginal likelihood estimator** — variance grows with
  the length of the schedule; adaptive schedules mitigate.

## Related in this repo

- `particle-filter-smc`, `parallel-tempering-mcmc`,
  `bouncy-particle-sampler`, `zig-zag-sampler` — MCMC and
  particle-method neighbours.
- `bridge-sampling-evidence`, `nested-sampling`,
  `bayesian-model-comparison` — alternatives for marginal
  likelihoods.
- `abc-approximate-bayesian`, `variational-inference` —
  approximate posterior samplers.
- `importance-sampling`, `sequential-analysis` — foundational
  ingredients.

## Run

```
python techniques/smc-samplers-del-moral/python/smc_samplers_del_moral.py
Rscript techniques/smc-samplers-del-moral/r/smc_samplers_del_moral.R
```

**Refs:** Del Moral, P., Doucet, A. and Jasra, A. "Sequential Monte Carlo samplers." *J. R. Stat. Soc. B*, 68(3): 411-436, 2006.

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
