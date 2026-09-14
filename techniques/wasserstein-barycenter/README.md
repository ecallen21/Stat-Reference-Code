# Wasserstein Barycenter (Reference §47.313)

Agueh & Carlier (2011); Cuturi & Doucet (2014). Given
probability measures `μ_1, …, μ_K` on the same space, their
(regularised) barycenter minimises the weighted sum of
Wasserstein distances:

```
ν* = argmin_ν Σ_k w_k · W_p^p(ν, μ_k)
```

Cuturi-Doucet use entropy-regularised Sinkhorn for efficient
approximate barycenters.

## Files

- `python/wasserstein_barycenter.py` — 1-D fixed-support
  barycenter of two Gaussians (means ±2) via Sinkhorn.
  Sweeping weights (0.5, 0.5), (0.2, 0.8), (0.8, 0.2) shifts
  the barycenter's mean in the expected direction (regularised
  Sinkhorn under-shoots slightly due to entropy smoothing).
- `r/wasserstein_barycenter.R` — `T4transport`, `transport`
  (R); POT.bregman.barycenter, `geomloss`, from-scratch
  (Python).

## When to use

- **Averaging distributions** in a shape-aware way — climate
  ensembles, sensor fusion, generative-model interpolation.
- **Image / shape morphing** — smooth interpolation between
  discrete measures.
- **Statistical registration** — align distributions that
  differ by translation.

## When NOT to use

- **When distributions live on very different supports** —
  Wasserstein needs a shared ground metric.
- **When linear averaging is what you want** — L2 average is
  simpler.
- **Very high-dimensional distributions** — Sinkhorn scales
  poorly; use sliced-Wasserstein.

## Assumptions & caveats

- **Regularisation biases** — larger ε (reg) smooths
  more; small ε recovers true W_p but is slower / unstable.
- **Fixed vs free support** — free-support barycenter
  (Staib 2017) is more flexible but expensive.
- **Sinkhorn iterations** — 200-500 typical; log-domain
  implementation for stability.
- **Convergence** — barycenter iterations may need > 100
  outer steps for tight convergence.

## Related in this repo

- `sinkhorn-divergence` — the underlying OT tool.
- `optimal-transport-wasserstein` — general OT.
- `wgan-wasserstein-gan`, `wgan-gp-gradient-penalty` — GAN
  applications of W-1.

## Run

```
python techniques/wasserstein-barycenter/python/wasserstein_barycenter.py
Rscript techniques/wasserstein-barycenter/r/wasserstein_barycenter.R
```

**Refs:** Agueh, M. and Carlier, G. "Barycenters in the Wasserstein space." *SIAM J. Math. Anal.*, 43(2): 904-924, 2011; Cuturi, M. and Doucet, A. "Fast computation of Wasserstein barycenters." In *ICML*, pp. 685-693, 2014.

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
