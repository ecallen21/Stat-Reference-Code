# Information Geometry (Reference §34.15)

Amari (1985, 2016). A **parametric family** `{ p(x; θ) }` is a
Riemannian manifold with the **Fisher-Rao metric**
`g_ij(θ) = I(θ)_ij`.

## Two headline consequences

- **KL ≈ ½ Fisher quadratic**:
  `KL(θ ‖ θ + dθ) = ½ dθᵀ I(θ) dθ + o(‖dθ‖²)`.
- **Natural gradient** (Amari 1998): pre-condition steepest descent
  by `I(θ)⁻¹`. Invariant to reparameterisation; equal to Newton's
  method for exponential families.

## When to use

- **Optimising over probability distributions** — VI, RL policy
  optimisation (NPG, TRPO, PPO), Bayesian NNs.
- **Reparameterisation-invariant gradients** — natural gradient.
- **Understanding curvature of statistical models** — where
  parameters are "close" or "far".

## When NOT to use

- **Large `p`** — computing `I⁻¹` is `O(p³)`; use K-FAC / Fisher-
  block approximations.
- **Non-parametric models** — the manifold is infinite-dim; use
  functional gradient or kernel methods.

## Files

- `python/information_geometry.py` —
  1. Verify KL ≈ ½ Fisher-quadratic for Gaussian family across `|dθ|
     ∈ {0.1, 0.05, 0.01}`.
  2. Natural vs vanilla gradient on a logistic MLE. **Natural
     converges in 4 iterations; vanilla still not at optimum after
     50**.
- `r/information_geometry.R` — `geomstats` (Python), community
  implementations for R.

## Assumptions & caveats

- **Regularity of the family** — Fisher info must exist and be
  well-defined.
- **Numerical stability** — `I(θ)` may be near-singular; add ridge
  `I + εI` before inversion.
- **Large-parameter approximations** — K-FAC (Martens-Grosse 2015)
  block-diagonalises Fisher for neural nets.
- **Exponential families** — natural parameters + expectation
  parameters form the two dual coordinate systems (α = ±1 in Amari's
  ±1-connections).

## Related in this repo

- `fisher-information` — the metric-tensor building block.
- `information-criteria`, `kl-divergence`, `shannon-entropy` — sister
  info-theoretic quantities.
- `bayesian-neural-network` — natural gradient shows up in VI training.
- `sgd-momentum`, `adam-optimizer` — the "flat-space" alternatives.

## Run

```
python techniques/information-geometry/python/information_geometry.py
Rscript techniques/information-geometry/r/information_geometry.R
```

**Refs:** Amari, S.-I. *Information Geometry and Its Applications*, Springer, 2016; Amari, S.-I. "Natural gradient works efficiently in learning." *Neural Computation*, 1998; Martens, J. & Grosse, R. "Optimizing neural networks with Kronecker-factored approximate curvature (K-FAC)." *ICML*, 2015.

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
