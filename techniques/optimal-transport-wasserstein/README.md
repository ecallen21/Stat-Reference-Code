# Optimal Transport / Wasserstein Distance (Reference §46.16)

Monge (1781); Kantorovich (1942); Cuturi (2013, Sinkhorn). Distance
between probability measures via the **minimum cost to move mass**.

## Definitions

    W_p(α, β)^p  =  inf_{γ ∈ Coupling(α, β)}  ∫ d(x, y)^p dγ(x, y)

Discrete atoms:

    W_1 = min_{T ≥ 0}  Σᵢⱼ Tᵢⱼ · d(xᵢ, yⱼ)
    s.t. Σⱼ Tᵢⱼ = αᵢ  and  Σᵢ Tᵢⱼ = βⱼ

Entropic-regularised (Sinkhorn):

    W_ε = min_T  Σ T · d + ε · Σ T · (log T − 1)

## Files

- `python/optimal_transport_wasserstein.py` — exact 1-D W₁ via
  sorted-CDF trick + Sinkhorn (iterative scaling) for n-D from
  scratch. Demo: 1-D W₁(N(0,1), N(1,1)) = 0.958 (analytic 1.000);
  W₁(N(0,1), N(0,2)) = 0.873 (analytic √(2/π) = 0.798); 2-D Sinkhorn
  on n=20 point clouds = 0.759, marginals within numerical
  tolerance.
- `r/optimal_transport_wasserstein.R` — `transport::wasserstein`,
  `transport::sinkhornTransport`, `approxOT` (R); POT, geomloss,
  `scipy.stats.wasserstein_distance`, ott-jax (Python).

## When to use

- **Distributional comparison** — data drift monitoring, generative
  model evaluation (FID, Sinkhorn divergence).
- **Domain adaptation** — align source & target feature distributions.
- **Sample alignment** — matching color histograms, point clouds,
  particle configurations.
- **Generative models** — Wasserstein GAN, flow-based models.

## When NOT to use

- **KL / f-divergences suffice** — if measures share support and you
  just need a divergence.
- **Very high dimensions with few samples** — W becomes noisy; use
  sliced-W or Sinkhorn with entropic regularisation.
- **You need a smooth metric** — Sinkhorn divergences are the
  differentiable alternative for deep-learning use.

## Assumptions & caveats

- **Ground metric `d`** — must reflect the true cost you care about.
- **Non-negative measures with equal mass** — for probability
  distributions; unbalanced OT relaxes this.
- **Cost of exact solver** `O(n³ log n)` for network-simplex;
  Sinkhorn is `O(n²)` per iteration with regulariser bias.
- **Regularisation bias** — Sinkhorn W is a **biased** approximation
  of the true W; use Sinkhorn *divergence* `S(α, β) = W_ε(α, β) −
  ½ W_ε(α, α) − ½ W_ε(β, β)` for zero-diagonal.

## Related in this repo

- `f-divergences`, `kl-divergence`, `total-variation` (see
  divergences) — alternative distances.
- `data-drift-detection`, `covariate-shift-adaptation` — canonical
  applications.
- `distributionally-robust-optimization` — Wasserstein-DRO builds on
  it.
- `gan-training`, `diffusion-model` — generative-model losses often
  Sinkhorn-based.

## Run

```
python techniques/optimal-transport-wasserstein/python/optimal_transport_wasserstein.py
Rscript techniques/optimal-transport-wasserstein/r/optimal_transport_wasserstein.R
```

**Refs:** Monge, G. *Mémoire sur la théorie des déblais et des remblais*, 1781; Kantorovich, L.V. "On the translocation of masses." *Dokl. Akad. Nauk SSSR*, 37, 1942; Cuturi, M. "Sinkhorn distances: lightspeed computation of optimal transport." *NIPS*, 2013.

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
