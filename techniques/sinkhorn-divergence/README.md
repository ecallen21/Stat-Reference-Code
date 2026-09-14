# Sinkhorn Divergence (Reference §47.314)

Cuturi (2013, Sinkhorn OT); Feydy et al (2019, de-biasing).
Entropy-regularised optimal transport:

```
OT_ε(μ, ν) = min_T ⟨T, C⟩ + ε · H(T)      s.t. row/col sums
```

De-biased Sinkhorn divergence:

```
S_ε(μ, ν) = OT_ε(μ, ν) − ½ (OT_ε(μ, μ) + OT_ε(ν, ν))
```

Positive and zero iff μ = ν, unlike the vanilla OT_ε.

## Files

- `python/sinkhorn_divergence.py` — Sinkhorn iterates for
  1-D discrete distributions. Demo: OT_ε(μ, μ) > 0 due to
  entropy (0.09 at ε=0.1); S_ε(μ, μ) ≈ 0 (de-biased). Across
  shifts 0.5 / 1.5 / 3.0, S_ε tracks the true squared W2
  qualitatively.
- `r/sinkhorn_divergence.R` — `T4transport`,
  `transport::sinkhorn`, reticulate + `geomloss` (R); POT
  sinkhorn / `geomloss.SamplesLoss`, from-scratch (Python).

## When to use

- **Differentiable OT loss** in generative modelling
  (Sinkhorn-GAN, cost-training).
- **Distribution matching** in domain adaptation.
- **Fast approximate Wasserstein** distance for large N.

## When NOT to use

- **Very small ε** — Sinkhorn iterations become unstable;
  use log-domain implementation.
- **Extremely high-dimensional support** — need sliced-Wasserstein
  or MMD.
- **Discrete-with-no-common-support** — OT ill-defined
  without regularisation.

## Assumptions & caveats

- **ε trade-off** — smaller ε → tighter Wasserstein
  approximation but more iterations / numerical instability.
- **De-biasing critical** — vanilla OT_ε with ε > 0 has
  OT_ε(μ, μ) > 0; do NOT use for MMD-style tests.
- **Convergence rate** — Sinkhorn is O(1/n) per row/col
  update; ~100-500 iterations typical.
- **Gradient computation** — auto-differentiation through
  Sinkhorn scales with iteration count.

## Related in this repo

- `wasserstein-barycenter` — uses Sinkhorn internally.
- `optimal-transport-wasserstein`, `wgan-wasserstein-gan`,
  `wgan-gp-gradient-penalty` — related OT applications.
- `f-divergences`, `kl-divergence`, `information-geometry`
  — alternative distances between distributions.

## Run

```
python techniques/sinkhorn-divergence/python/sinkhorn_divergence.py
Rscript techniques/sinkhorn-divergence/r/sinkhorn_divergence.R
```

**Refs:** Cuturi, M. "Sinkhorn distances: Lightspeed computation of optimal transport." In *NeurIPS*, 2013; Feydy, J. et al. "Interpolating between optimal transport and MMD using Sinkhorn divergences." In *AISTATS*, 2019.

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
