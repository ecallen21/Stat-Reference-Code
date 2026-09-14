# Differential Evolution (Reference §47.377)

Storn & Price (1997). Population-based evolutionary optimiser.
For each individual `x_i` in the population, sample three
DISTINCT others `a, b, c`, form a mutant `v = a + F · (b − c)`,
apply binomial crossover with rate `CR` against `x_i` to make
`u`, and accept `u` if `f(u) < f(x_i)`:

```
for each x_i in population:
    a, b, c = 3 distinct others
    v = a + F · (b − c)             (mutation)
    u = crossover(x_i, v, CR)
    if f(u) < f(x_i): replace x_i with u
```

Simple, gradient-free, competitive with CMA-ES / genetic
algorithms on many benchmark suites. Reliable global-optimiser
choice when only function evaluations are available.

## Files

- `python/differential_evolution.py` — Rastrigin function in
  d=2, 5, 10 (multi-modal, global minimum 0.0). Achieves
  `f = 0.00` in d=2, `5.02` in d=5, `42.4` in d=10 (300
  gens, pop 30). `scipy.optimize.differential_evolution`
  gets 0.0, 0.0, 3.98 on the same benchmark.
- `r/differential_evolution.R` — `DEoptim::DEoptim`,
  `RcppDE` (R); `scipy.optimize.differential_evolution`,
  from-scratch (Python).

## When to use

- **Non-differentiable, noisy, or black-box objectives** —
  global search on Rastrigin-like landscapes.
- **Hyperparameter tuning** — competitive with Bayesian
  optimisation for moderate budgets.
- **Small-to-medium dim (< 50)** — beyond that, CMA-ES /
  ES-1+1 with covariance updates usually win.
- **When you need a reliable baseline** with few knobs.

## When NOT to use

- **Smooth convex** — L-BFGS / Newton crushes DE.
- **Very expensive objectives** — Bayesian optimisation is
  more sample-efficient.
- **Discrete / combinatorial problems** — GA / simulated
  annealing with tailored operators is better.

## Assumptions & caveats

- **F (mutation strength)** — 0.5-0.9 typical; too small
  ⇒ premature convergence, too large ⇒ divergence.
- **CR (crossover rate)** — 0.9 for separable, 0.1 for non-
  separable.
- **Strategies** — DE/rand/1/bin (default), DE/best/2/bin,
  DE/current-to-best/1/bin; adaptive variants (JADE, SHADE)
  auto-tune F and CR.
- **Boundary handling** — clip or reflect at box bounds.
- **Random seeding** — reproducibility only under fixed seed;
  DE is stochastic.

## Related in this repo

- `cma-es-evolution-strategy`, `genetic-algorithm`,
  `particle-swarm-optimization`, `cross-entropy-method`,
  `basin-hopping` — evolutionary / global-optimisation
  neighbours.
- `simulated-annealing`, `bayesian-optimization` — other
  black-box global optimisers.
- `nelder-mead-simplex` — derivative-free local method.
- `hyperband-multi-fidelity`, `bohb-bayesian-hyperband` —
  hyperparameter-optimisation cousins.

## Run

```
python techniques/differential-evolution/python/differential_evolution.py
Rscript techniques/differential-evolution/r/differential_evolution.R
```

**Refs:** Storn, R. and Price, K. "Differential evolution — a simple and efficient heuristic for global optimization over continuous spaces." *J. Glob. Optim.*, 11(4): 341-359, 1997.

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
