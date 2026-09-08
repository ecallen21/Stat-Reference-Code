# Genetic Algorithm (Reference §47.141)

Holland (1975); Goldberg (1989). Population-based search using
biology-inspired operators:

    1. Rank / tournament / roulette selection of parents.
    2. Crossover (single-point / uniform / SBX) → offspring.
    3. Mutation (bit-flip / Gaussian) of offspring.
    4. Repeat over generations; keep the best solution.

Naturally handles discrete / combinatorial / mixed spaces where
gradient methods cannot apply.

## Files

- `python/genetic_algorithm.py` — binary-string GA with tournament
  selection (k = 3), 1-point crossover (p = 0.8), bit-flip mutation
  (p = 0.03). 0/1 knapsack with n = 30 items, capacity = 40 % of
  total weight:
  - 20 gens: value = 292 / weight 125 (14 items)
  - 100 gens: value = 308 / weight 126 (15 items)
  - **500 gens: value = 310 / weight 126** — beats greedy
    value/weight baseline (305).
- `r/genetic_algorithm.R` — `GA::ga` (binary / real-valued /
  permutation); `DEAP` / `pymoo` in Python.

## When to use

- **Discrete / combinatorial** optimisation (knapsack, feature
  subset selection, scheduling, permutation problems).
- **Multi-objective** with NSGA-II variants.
- **Black-box** objectives with mixed variable types.

## When NOT to use

- **Smooth continuous** objectives — CMA-ES / L-BFGS are faster and
  more accurate.
- **Very tight computational budget** — GA needs many evaluations.
- **When exact algorithms exist** (LP, branch-and-bound for
  well-structured problems).

## Assumptions & caveats

- **Chromosome encoding** matters — binary vs real vs
  permutation crossovers differ substantially.
- **Elitism** (carry-over best-of-generation) improves convergence
  reliability.
- **Diversity collapse** is the usual failure mode; large
  population, higher mutation, or restart helps.
- **Penalty vs repair** for infeasible offspring — knapsack demo
  here uses a linear penalty on weight overflow.

## Related in this repo

- `simulated-annealing` — single-particle analogue for
  combinatorial problems.
- `particle-swarm-optimization`,
  `cma-es-evolution-strategy`, `evolution-strategies-openai` —
  sibling metaheuristics.
- `bayesian-optimization` — sample-efficient alternative for
  expensive f.
- `nsga-multi-objective` (in principle) — multi-objective GA
  variant.

## Run

```
python techniques/genetic-algorithm/python/genetic_algorithm.py
Rscript techniques/genetic-algorithm/r/genetic_algorithm.R
```

**Refs:** Holland, J. H. *Adaptation in Natural and Artificial Systems*, U Michigan Press, 1975; Goldberg, D. E. *Genetic Algorithms in Search, Optimization and Machine Learning*, Addison-Wesley, 1989; Scrucca, L. "GA: A package for genetic algorithms in R." *J Stat Softw* 53(4), 2013.

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
