# Evolution Strategies — OpenAI variant (Reference §47.147)

Salimans, Ho, Chen, Sidor & Sutskever (2017). Antithetic
finite-difference gradient estimator on a Gaussian-smoothed
objective F(θ) = E_{ε ~ N(0, I)}[f(θ + σ ε)]:

    ∇_θ F ≈ (1 / (n σ)) Σ_i f(θ + σ ε_i) ε_i,      ε_i antithetic.

Uses **rank-shaping** (map rewards to centered ranks in [-0.5,
0.5]) to make the update scale-free. Highly parallelisable — each
worker needs only its seed and a scalar return.

## Files

- `python/evolution_strategies_openai.py` — from-scratch OpenAI ES
  with antithetic pairs + rank scores + fixed sigma / lr.
  - **10-D quadratic bowl**, x0 = 3·1: 300 iters converge to
    f = 1.4e-4, ‖x‖∞ = 0.006.
  - **5-D step function** ⌊x⌋² (non-differentiable everywhere):
    already at 50 iters ES converges to ⌊x⌋ = 0 (f = 0), which
    gradient methods cannot touch at all.
- `r/evolution_strategies_openai.R` — no CRAN package; a minimal
  from-scratch R port of the same recipe on the 10-D bowl.

## When to use

- **Deep-RL policy search** where gradients are noisy /
  hard-to-estimate (Salimans et al. beat A3C on Atari with 1440
  CPU-hours vs 1 GPU-day).
- **Non-differentiable / piecewise-constant / discontinuous**
  objectives.
- **Massively parallel** clusters — trivial data parallelism, only
  seeds + scalar returns are exchanged.

## When NOT to use

- **Smooth, low-dim** problems — L-BFGS / Adam are far cheaper.
- **Sample-efficient** regimes — ES needs thousands of forward
  evaluations for even moderate d.
- **Very high dim without population scaling** — O(d) sample
  complexity per iter.

## Assumptions & caveats

- **σ (perturbation)** — too small → poor gradient estimate; too
  large → bias. Fixed σ works surprisingly well; some variants
  adapt σ (like CMA-ES).
- **Rank shaping** is crucial for scale robustness (raw rewards
  can dominate the gradient).
- **Antithetic sampling** halves the number of independent
  evaluations while preserving unbiasedness.
- **No credit assignment across time** — each policy is scored on
  a whole rollout return.

## Related in this repo

- `cma-es-evolution-strategy` — full-covariance / adaptive-σ
  cousin (usually more sample-efficient for continuous
  optimisation).
- `particle-swarm-optimization`, `genetic-algorithm`,
  `simulated-annealing` — sibling metaheuristics.
- `ppo-clipped`, `sac-soft-actor-critic`, `rainbow-dqn` — RL
  algorithms ES competes with for deep-RL benchmarks.

## Run

```
python techniques/evolution-strategies-openai/python/evolution_strategies_openai.py
Rscript techniques/evolution-strategies-openai/r/evolution_strategies_openai.R
```

**Refs:** Salimans, T., Ho, J., Chen, X., Sidor, S. & Sutskever, I. "Evolution strategies as a scalable alternative to reinforcement learning." *arXiv:1703.03864*, 2017; Wierstra, D. et al. "Natural evolution strategies." *JMLR* 15(1), 2014.

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
