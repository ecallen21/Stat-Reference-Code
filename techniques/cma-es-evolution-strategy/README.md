# CMA-ES — Covariance Matrix Adaptation ES (Reference §47.138)

Hansen & Ostermeier (2001). Population-based, derivative-free
optimiser that adapts a full-covariance multivariate-Gaussian search
distribution to the local landscape. Each generation:

    x_i ~ mean + sigma * N(0, C),   i = 1..lambda
    rank offspring by f, keep top mu
    update  mean, C (rank-1 + rank-mu), sigma (path-length control).

Considered the state of the art for continuous black-box
optimisation with modest dimension (up to a few hundred).

## Files

- `python/cma_es_evolution_strategy.py` — from-scratch CMA-ES with
  evolution paths, rank-1 + rank-mu covariance update, path-length
  step-size control. 5-D Rosenbrock, x0 ~ Uniform(-2, 2):
  - 50 iters: f_best ≈ 4.2   ‖x-1‖∞ ≈ 1.5
  - 200 iters: f_best ≈ 3.6e-5   ‖x-1‖∞ ≈ 3e-3
  - 500 iters: f_best = **0.000000**   ‖x-1‖∞ = **0.0000** — recovers
    the optimum x = 1.
- `r/cma_es_evolution_strategy.R` — `cmaes::cma_es`; alternative
  `cmaesr`, `adagio::pureCMAES`.

## When to use

- **Continuous, black-box objective** with dimension d ≤ ~100.
- **Non-differentiable / noisy / rugged** landscapes where gradient
  methods fail.
- **Hyperparameter tuning** for small-to-medium search spaces.

## When NOT to use

- **Differentiable** objectives — L-BFGS / Adam are far faster.
- **Discrete / combinatorial** spaces — use GA, simulated
  annealing.
- **Very high-dimensional** (d ≥ 1000) — sep-CMA-ES / LM-CMA
  variants exist but O(d²) covariance updates get costly.

## Assumptions & caveats

- **Population size** lambda = 4 + ⌊3 ln d⌋ default; increase for
  multimodal landscapes.
- **Restart CMA-ES** (IPOP / BIPOP) handles multimodality by
  doubling population on restart.
- **Bound constraints** need box-transform or reflection; equality
  constraints need Lagrangian variants.

## Related in this repo

- `particle-swarm-optimization`, `genetic-algorithm`,
  `evolution-strategies-openai` — sibling population-based
  optimisers.
- `simulated-annealing` — single-particle Metropolis analogue.
- `bayesian-optimization`, `random-search` — sample-efficient
  alternatives for expensive f.

## Run

```
python techniques/cma-es-evolution-strategy/python/cma_es_evolution_strategy.py
Rscript techniques/cma-es-evolution-strategy/r/cma_es_evolution_strategy.R
```

**Refs:** Hansen, N. & Ostermeier, A. "Completely derandomized self-adaptation in evolution strategies." *Evolutionary Computation* 9(2), 2001; Hansen, N. "The CMA Evolution Strategy: A tutorial." *arXiv:1604.00772*, 2016.

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
