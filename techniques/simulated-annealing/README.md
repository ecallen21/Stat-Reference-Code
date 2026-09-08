# Simulated Annealing (Reference §47.140)

Kirkpatrick, Gelatt & Vecchi (1983). Metropolis-based global
optimiser modelled on physical annealing of metals:

    propose x' ~ q(·|x)
    accept if f(x') < f(x); else accept w.p.  exp(-(f(x')-f(x))/T)
    cool  T_k = T_0 * alpha^k.

Converges in probability to the global optimum under logarithmic
cooling; geometric cooling (α ≈ 0.99–0.9999) is the practical
default. Handles discrete + continuous variables uniformly.

## Files

- `python/simulated_annealing.py` — two demos:
  1. **Continuous 1-D multi-well** f(x) = x² + 20 sin²(x). Starting
     x = -3, SA (T0=5, α=0.9995, step=1.0, 20k iters) finds
     x* = **0.000**, f* ≈ 0.0000 — escapes the local minima at
     x ≈ -3 (f ≈ 9) and x ≈ -π (f ≈ 9.9).
  2. **15-city TSP** via 2-opt neighbours. SA tour length = **3.77**
     vs random-permutation avg baseline = 8.48 (~55 % improvement).
- `r/simulated_annealing.R` — `GenSA::GenSA` (generalised SA);
  `optim(method="SANN")` for a bare-bones version.

## When to use

- **Discrete combinatorial** problems (TSP, VRP, scheduling,
  bin-packing).
- **Rugged, multimodal continuous** landscapes.
- **Constrained optimisation** where feasibility is easy to preserve
  in the proposal kernel.

## When NOT to use

- **Smooth, differentiable** objectives — L-BFGS / Adam are orders
  faster.
- **Very tight budgets** — SA needs many function evaluations to
  cool properly.
- **Highly noisy f** — noise gets amplified by Metropolis accept /
  reject; use ES / CMA-ES.

## Assumptions & caveats

- **Initial temperature T_0** must exceed typical Δf so the walker
  starts near-uniform.
- **Cooling schedule** determines convergence guarantee vs speed:
  log(k+1) is theoretical, α^k is practical.
- **Proposal kernel** should be tuned so ~30-50 % of moves accept.
- **Restart from multiple seeds** for tough multimodal problems.

## Related in this repo

- `metropolis-hastings-mcmc` — same accept rule, different
  purpose (sampling vs optimising).
- `genetic-algorithm`, `particle-swarm-optimization`,
  `cma-es-evolution-strategy` — population-based cousins.
- `mcmc-tsp` variants for combinatorial sampling.

## Run

```
python techniques/simulated-annealing/python/simulated_annealing.py
Rscript techniques/simulated-annealing/r/simulated_annealing.R
```

**Refs:** Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P. "Optimization by simulated annealing." *Science* 220(4598), 1983; Metropolis, N. et al. "Equation of state calculations by fast computing machines." *J Chem Phys* 21(6), 1953; Xiang, Y., Gubian, S., Suomela, B. & Hoeng, J. "Generalized simulated annealing for global optimization: the GenSA package." *R Journal* 5(1), 2013.

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
