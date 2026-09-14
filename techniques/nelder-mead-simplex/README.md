# Nelder-Mead Simplex (Reference §47.344)

Nelder & Mead (1965). Derivative-free minimisation on `ℝⁿ`
using a moving simplex of `n + 1` vertices; each iteration
reflects, expands, contracts, or shrinks the worst vertex
using coefficients `α = 1, γ = 2, ρ = 0.5, σ = 0.5`.

## Files

- `python/nelder_mead_simplex.py` — Rosenbrock in 2D (145
  iters, `f* = 2e-20`) and 5D (631 iters, `f* = 4e-19`).
  Matches `scipy.optimize.minimize(method='Nelder-Mead')`
  to machine precision.
- `r/nelder_mead_simplex.R` — `optim(method='Nelder-Mead')`,
  `dfoptim::nmk` (R); `scipy.optimize.minimize`, from-scratch
  (Python).

## When to use

- **Objective with no gradient** — black-box, noisy, or
  simulation-based `f`.
- **Low-dimensional** (n ≤ 20) — cost per iteration is O(n),
  but curse of dimensionality kicks in.
- **Prototyping** — good default when smoothness / gradients
  are unknown.

## When NOT to use

- **High-dimensional (n ≫ 20)** — simplex collapses; use
  CMA-ES, BOBYQA, or gradient-based if AD available.
- **Stochastic objectives** with high variance — simplex
  ordering is unreliable.
- **Constrained problems** — vanilla NM ignores bounds; use
  Powell's BOBYQA, `dfoptim::nmk`, or penalty methods.

## Assumptions & caveats

- **Local method** — converges to a local minimum; multi-
  start recommended.
- **No convergence proof** — McKinnon (1998) found simple
  cases where NM stalls; monitor simplex volume.
- **Adaptive parameters** — Gao-Han (2012) modify (α,γ,ρ,σ)
  by dimension.
- **Cauchy iterate** — the centroid path is a Cauchy
  sequence when NM works; test with function-value tolerance
  AND simplex spread.

## Related in this repo

- `cma-es-evolution-strategy` — modern derivative-free
  optimiser, dominates in medium/high dim.
- `simulated-annealing`, `genetic-algorithm` — stochastic
  black-box alternatives.
- `bayesian-optimization` — sample-efficient for expensive
  objectives.
- `particle-swarm-optimization` — population-based cousin.

## Run

```
python techniques/nelder-mead-simplex/python/nelder_mead_simplex.py
Rscript techniques/nelder-mead-simplex/r/nelder_mead_simplex.R
```

**Refs:** Nelder, J.A. and Mead, R. "A simplex method for function minimization." *Comput. J.*, 7(4): 308-313, 1965.

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
