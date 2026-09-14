# Basin Hopping (Reference §47.378)

Wales & Doye (1997). Global optimisation by iterating:

1. **Local minimisation** from the current point.
2. **Random perturbation** `x' = x + N(0, step² I)`.
3. **Metropolis acceptance** — accept `x'` if `f(x') < f(x)`
   or with probability `exp(−ΔF / T)`.

The state space is compressed into "basins" (local minima)
and the random hop lets the algorithm cross energy barriers.
Very effective on rugged energy landscapes: protein folding,
atomic cluster minimisation (Wales-Doye's original target),
multi-modal statistical objectives.

## Files

- `python/basin_hopping.py` — Multi-modal 2D function
  `(x−1)²(x+1)² + y² + 5 sin(3x) sin(3y)` reaches
  `f = −4.24`. Rastrigin d=5 reaches `f ≈ 1.99` (nearly the
  global optimum 0; the last two coordinates hit `±0.995`).
  Uses `scipy.optimize.minimize(Nelder-Mead)` as the inner
  local optimiser.
- `r/basin_hopping.R` — `GenSA::GenSA` (generalised simulated
  annealing, related hybrid), custom loop (R);
  `scipy.optimize.basinhopping`, from-scratch (Python).

## When to use

- **Rugged, multi-modal objectives** — cluster minimisation
  in chemistry / physics; energy-landscape problems.
- **When each local minimum is easy to reach** but the number
  of basins is huge.
- **Hybrid with any local optimiser** — L-BFGS, Nelder-Mead,
  trust-region all fit.

## When NOT to use

- **Smooth unimodal problems** — plain L-BFGS wins.
- **Very expensive functions** — Bayesian optimisation is
  more sample-efficient.
- **Combinatorial spaces** — simulated annealing / GA with
  discrete moves is more natural.

## Assumptions & caveats

- **Step size** — matched to basin width; too small ⇒ trapped,
  too large ⇒ wasteful.
- **Temperature T** — controls Metropolis acceptance; standard
  is to keep T fixed since the local step already reduces
  the objective.
- **Local optimiser** — Nelder-Mead / L-BFGS-B typical; the
  hop's random direction takes you between basins, the local
  opt slides you down.
- **Stopping** — usually a fixed iteration budget; no
  gradient-based convergence.
- **Not to confuse with simulated annealing** — SA does not
  local-min after each hop, so it explores the *full*
  landscape; basin hopping only visits basin bottoms.

## Related in this repo

- `differential-evolution`, `cma-es-evolution-strategy`,
  `genetic-algorithm`, `simulated-annealing`,
  `particle-swarm-optimization`, `cross-entropy-method` —
  global-optimisation neighbours.
- `nelder-mead-simplex`, `lbfgs-quasi-newton`,
  `trust-region-optimization` — inner local optimisers.
- `bayesian-optimization` — sample-efficient global cousin.

## Run

```
python techniques/basin-hopping/python/basin_hopping.py
Rscript techniques/basin-hopping/r/basin_hopping.R
```

**Refs:** Wales, D.J. and Doye, J.P.K. "Global optimization by basin-hopping and the lowest energy structures of Lennard-Jones clusters containing up to 110 atoms." *J. Phys. Chem. A*, 101: 5111-5116, 1997.

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
