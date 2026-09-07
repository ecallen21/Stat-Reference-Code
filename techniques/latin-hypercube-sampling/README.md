# Latin Hypercube Sampling (Reference §45.7)

McKay-Beckman-Conover (1979), Morris & Mitchell (1995). Stratified
sampling for computer-experiment design: divide each of `d`
dimensions into `n` equal-probability bins, then permute across
dimensions so each bin appears exactly once per dimension.

## Variants

- **Standard LHS** — random permutation within each dimension.
- **Maximin LHS** — try many realisations; keep the one that
  maximises minimum pairwise distance (Morris-Mitchell).
- **Orthogonal LHS** — extra orthogonality between columns.
- **Sliced LHS** — hierarchical LHS for multi-fidelity codes.

## When to use

- **Emulator training** — inputs to expensive simulator +
  Gaussian-process / kriging surrogate.
- **Sensitivity analysis** — Sobol / Morris designs use LHS as a
  starting point.
- **Screening** — better space-filling than random with the same n.

## When NOT to use

- **Estimation problems** with statistical inference — random
  sampling gives standard theory; LHS needs replication-friendly
  variance estimators (McKay-Beckman-Conover).

## Files

- `python/latin_hypercube_sampling.py` — standard LHS + Maximin
  LHS (custom). Demo (10 pts × 3 dims): every bin covered per
  column; **30 pts × 5 dims**: maximin-LHS min distance **0.325**
  vs plain random **0.230** — better space-filling with the same
  sample size.
- `r/latin_hypercube_sampling.R` — `lhs::randomLHS` /
  `maximinLHS`, `DiceDesign`, `SLHD` (R);
  `scipy.stats.qmc.LatinHypercube`, `pyDOE2`, `smt` (Python).

## Assumptions & caveats

- **1-D marginals uniform** by construction — 2-D projections need
  the maximin criterion to actually space-fill.
- **Random permutation** may create diagonal patterns; maximin
  variants reduce that.
- **Optimality is heuristic** — global maximin design is NP-hard;
  iterative Morris-Mitchell / simulated annealing helps.
- **Report the design** (seed, method) so downstream results are
  reproducible.

## Related in this repo

- `fractional-factorial`, `response-surface`, `d-optimal-design`,
  `taguchi-methods` — companion DOE tools.
- `importance-sampling`, `mcmc-metropolis-hastings` — alternative
  sampling primitives.

## Run

```
python techniques/latin-hypercube-sampling/python/latin_hypercube_sampling.py
Rscript techniques/latin-hypercube-sampling/r/latin_hypercube_sampling.R
```

**Refs:** McKay, M.D., Beckman, R.J., & Conover, W.J. "A comparison of three methods for selecting values of input variables in the analysis of output from a computer code." *Technometrics*, 1979; Morris, M.D. & Mitchell, T.J. "Exploratory designs for computational experiments." *Journal of Statistical Planning and Inference*, 1995.

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
