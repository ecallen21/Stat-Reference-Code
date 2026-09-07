# Nested Sampling (Reference §25.10)

Skilling (2006). Estimates the Bayesian **evidence** `Z = ∫ L(θ) π(θ) dθ`
by transforming the multi-dim integral into a 1-D integral of `L`
against prior mass `X ∈ (0, 1]`:

    Z = ∫₀¹ L(X) dX,       X_i ≈ exp(−i / N)

## Algorithm

1. Sample `N` "live points" from the prior; find the worst (lowest `L`).
2. Record its likelihood `L_i` and prior-mass estimate `X_i`.
3. Replace with a new sample from the prior **constrained to `L > L_i`**.
4. Repeat until `L` stops growing.
5. `Ẑ = Σᵢ (X_{i−1} − X_i) · L_i`.

## Files

- `python/nested_sampling.py` — rejection-based nested sampler from
  scratch on a 1-D Gaussian likelihood with uniform prior. Demo
  (true Z = 0.1): recovers Ẑ = 0.103.
- `r/nested_sampling.R` — `nestedmodels`, `RNested`,
  `BayesianTools` (R); `dynesty`, `nestle`, `ultranest`,
  MultiNest / PolyChord (Python).

## When to use

- **Bayesian evidence / Bayes factors** — Z falls out for free.
- **Multimodal posteriors** — dynesty / MultiNest handle modes
  MCMC struggles with.
- **Model comparison** — Bayes-factor evidence ratios for competing
  models.
- **Cosmology / physics** — the dominant tool for parameter +
  model inference in the field.

## When NOT to use

- **High-dim (d > 30)** — vanilla NS gets expensive; use dynamic
  or MLFriends-style adaptive sampling.
- **Very peaky posteriors** — the constrained-prior sampler struggles
  to find better live points.
- **You only need posterior samples** — HMC/NUTS is usually
  faster.

## Assumptions & caveats

- **Constrained sampling** — the hardest step; rejection works only
  for smooth low-d targets; MCMC / slice within ellipsoid for higher
  dims (dynesty, PolyChord).
- **Termination criterion** — stop when `dlog(Z) < ε`; too early
  underestimates Z.
- **Evidence uncertainty** — the NS Z estimate has quantifiable
  variance ~ `sqrt(H / N)` where H is information.
- **Prior transform** — samplers work in unit cube; you supply
  `theta = prior_transform(u)` mapping `(0, 1)^d → prior support`.

## Related in this repo

- `bridge-sampling-evidence`, `bayesian-model-comparison`,
  `information-criteria` — evidence / model-choice cousins.
- `mcmc-metropolis-hastings`, `hmc-nuts`, `slice-sampler` — MCMC
  alternatives (posterior-only, no evidence).
- `variational-inference` — ELBO evidence approximation.
- `particle-filter-smc` — related sequential MC method.

## Run

```
python techniques/nested-sampling/python/nested_sampling.py
Rscript techniques/nested-sampling/r/nested_sampling.R
```

**Refs:** Skilling, J. "Nested sampling for general Bayesian computation." *Bayesian Analysis*, 1(4): 833-859, 2006; Handley, W.J., Hobson, M.P. & Lasenby, A.N. "PolyChord: nested sampling for cosmology." *MNRAS*, 450(1): L61-L65, 2015.

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
