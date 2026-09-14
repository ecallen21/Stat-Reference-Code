# NSGA-II — Non-dominated Sorting GA (Reference §47.348)

Deb, Pratap, Agarwal & Meyarivan (2002). Multi-objective
evolutionary optimisation that recovers the Pareto front in a
single run using:

1. **Fast non-dominated sorting** into fronts F₁, F₂, …
2. **Crowding distance** — density estimate per front
3. **(µ + λ) selection** favouring lower rank then higher
   crowd distance (diversity)
4. **SBX crossover + polynomial mutation** for real-valued
   variables

## Files

- `python/nsga_ii.py` — ZDT1 test problem (d=6, convex
  Pareto). 80 individuals × 100 generations. Mean L1
  distance from the recovered front to the true Pareto
  front `f₂ = 1 − √f₁` is 0.004; the front is well-spread
  from `f₁ ≈ 0` to `≈ 1`.
- `r/nsga_ii.R` — `nsga2R`, `mco`, `MaOEA` (R);
  `pymoo.algorithms.moo.nsga2`, DEAP, from-scratch (Python).

## When to use

- **Multi-objective optimisation** with 2-3 conflicting
  objectives (accuracy vs latency, cost vs risk).
- **Black-box or simulation objectives** with cheap
  evaluations.
- **When a Pareto front (not a single trade-off) is required**
  for downstream decision-making.

## When NOT to use

- **Many objectives (≥ 4)** — NSGA-II's non-dominated set
  swamps out; use NSGA-III or reference-point methods.
- **Very expensive objectives** — Bayesian optimisation
  (qEHVI, ParEGO) is more sample-efficient.
- **Single-objective problems** — CMA-ES / L-BFGS dominate.

## Assumptions & caveats

- **Complexity** — O(M N²) per generation, M objectives, N
  population; naive sort is the bottleneck in high M.
- **SBX distribution index η** — controls exploration
  (η ≈ 15 is standard for real problems).
- **Constraint handling** — Deb's constrained-dominance;
  epsilon-constraint / repair operators for hard bounds.
- **Reproducibility** — seed both crossover and mutation
  RNG.
- **Solution ranking** — Pareto-optimality is set-valued;
  downstream picks a single point by a scalarisation or a
  decision-maker.

## Related in this repo

- `genetic-algorithm`, `cma-es-evolution-strategy`,
  `evolution-strategies-openai`, `particle-swarm-optimization`
  — single-objective evolutionary siblings.
- `bayesian-optimization` — sample-efficient MOO cousin.
- `benefit-risk-mcda` — multi-criteria decision analysis
  downstream.
- `hyperband-multi-fidelity`, `bohb-bayesian-hyperband` —
  multi-fidelity HPO overlap.

## Run

```
python techniques/nsga-ii/python/nsga_ii.py
Rscript techniques/nsga-ii/r/nsga_ii.R
```

**Refs:** Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. "A fast and elitist multiobjective genetic algorithm: NSGA-II." *IEEE Trans. Evol. Comput.*, 6(2): 182-197, 2002.

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
