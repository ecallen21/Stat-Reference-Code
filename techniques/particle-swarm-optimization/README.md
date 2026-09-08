# Particle Swarm Optimization (Reference §47.139)

Kennedy & Eberhart (1995). Population-based derivative-free
optimiser inspired by flocking / schooling. Each particle keeps
position `x`, velocity `v`, personal-best `p`; the swarm keeps a
global-best `g`. Updates:

    v ← w v + c1 r1 (p − x) + c2 r2 (g − x)
    x ← x + v

Inertia weight `w` decays linearly from `w_max` → `w_min` (Shi &
Eberhart 1998) to trade exploration → exploitation.

## Files

- `python/particle_swarm_optimization.py` — from-scratch PSO with
  inertia decay + box clipping. 5-D Rastrigin (highly multimodal,
  minimum at x = 0):
  - 50 iters: f_best ≈ 2.99   ‖x‖∞ ≈ 0.05
  - 200 iters: f_best ≈ 0.99   ‖x‖∞ ≈ 0.07
  - 500 iters: f_best = **0.0000**   swarm converges to the global.
- `r/particle_swarm_optimization.R` — `pso::psoptim` (SPSO 2007);
  alternatives: `psoptim`, `metaheuristicOpt`.

## When to use

- **Continuous, black-box optimisation** with cheap function
  evaluations.
- **Multimodal** landscapes where gradient methods get stuck.
- **Real-time / embedded** optimisation — simple update, low memory.

## When NOT to use

- **Differentiable, well-conditioned** objectives — gradient methods
  win.
- **Very high-dimensional** (d ≥ 500) — swarm size and iterations
  scale poorly.
- **Discrete-only** problems (though hybrid discrete-PSO variants
  exist).

## Assumptions & caveats

- **Coefficient tuning** (c1, c2) matters; standard c1 = c2 = 1.5
  (or 2.05 with Clerc constriction).
- **Premature convergence** to a local minimum if inertia decays
  too fast or swarm too small.
- **Bounded search** — natural fit for box constraints via clipping.
- **Not gradient-free of hyperparameters** — needs w_max, w_min,
  c1, c2, swarm size tuned per problem.

## Related in this repo

- `cma-es-evolution-strategy`, `genetic-algorithm`,
  `evolution-strategies-openai` — sibling population-based
  optimisers.
- `simulated-annealing` — single-particle stochastic search.
- `bayesian-optimization` — sample-efficient for expensive f.

## Run

```
python techniques/particle-swarm-optimization/python/particle_swarm_optimization.py
Rscript techniques/particle-swarm-optimization/r/particle_swarm_optimization.R
```

**Refs:** Kennedy, J. & Eberhart, R. "Particle swarm optimization." *IEEE ICNN*, 1995; Shi, Y. & Eberhart, R. "A modified particle swarm optimizer." *IEEE ICEC*, 1998; Clerc, M. "The swarm and the queen." *IEEE CEC*, 1999.

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
