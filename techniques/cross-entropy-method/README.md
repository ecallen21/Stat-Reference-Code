# Cross-Entropy Method (Reference §47.345)

Rubinstein (1997, 1999). Iterative importance-sampling for
RARE-EVENT simulation and for OPTIMISATION:

```
sample X ~ p(θ_t)
select elite fraction ρ (top-scoring samples)
refit θ_{t+1} by MLE on elites
```

For continuous optimisation the family `p(θ)` is usually
Gaussian `N(μ, Σ)`; for combinatorial problems it is
categorical. CEM is the ancestor of CMA-ES, EDA, and the
CEM-RL policy-search family.

## Files

- `python/cross_entropy_method.py` — Sphere d=10 (best
  `f(x)` = 4e-6 in 30 iterations) and Rastrigin d=5, a
  strongly multimodal function (best `f(x)` = 0.000 with
  `x* = 0` in 100 iterations).
- `r/cross_entropy_method.R` — `CEoptim::CEoptim` (R);
  `torchrl CEM`, `pypi cma`, from-scratch (Python).

## When to use

- **Rare-event simulation** — quantile estimation, tail-
  probability importance sampling.
- **Black-box optimisation** — non-differentiable, noisy,
  simulation-in-the-loop.
- **Combinatorial problems** — TSP, max-cut, scheduling
  (with categorical family).
- **Policy search in RL** — CEM-based baselines are strong on
  MuJoCo tasks and easy to implement.

## When NOT to use

- **When gradients are cheap** — Adam / L-BFGS crush CEM on
  smooth objectives.
- **Very high-dim continuous** — CMA-ES with rank-µ covariance
  update is more sample-efficient.
- **Convex problems** — trivial with interior-point / SQP.

## Assumptions & caveats

- **Elite fraction ρ** — 5-20 % typical; too greedy (small ρ)
  causes premature convergence.
- **Variance floor** — add `ε I` to prevent premature
  collapse.
- **Smoothing** — Kroese-Rubinstein 2013 recommend
  `θ_{t+1} = α θ_elite + (1 − α) θ_t` for stability.
- **Parametric family choice** — Gaussian is diagonal by
  default; full covariance grows to CMA-ES.
- **Convergence** — provable for finite discrete state
  spaces; heuristic elsewhere.

## Related in this repo

- `cma-es-evolution-strategy` — full-covariance rank-µ update.
- `genetic-algorithm`, `evolution-strategies-openai`,
  `particle-swarm-optimization` — other population-based
  methods.
- `simulated-annealing`, `bayesian-optimization` — global
  black-box optimisation cousins.
- `evolution-strategies-openai` — REINFORCE-style variant.

## Run

```
python techniques/cross-entropy-method/python/cross_entropy_method.py
Rscript techniques/cross-entropy-method/r/cross_entropy_method.R
```

**Refs:** Rubinstein, R.Y. "Optimization of computer simulation models with rare events." *Eur. J. Oper. Res.*, 99: 89-112, 1997; Rubinstein, R.Y. and Kroese, D.P. *The Cross-Entropy Method*, Springer, 2004.

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
