# Hyperband Multi-Fidelity Search (Reference §47.275)

Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar (2018).
Bandit-based hyperparameter optimisation that adaptively
allocates budget between exploration (many configs, low
budget) and exploitation (few, high budget). Runs successive-
halving brackets across geometrically decreasing `s`.

## Files

- `python/hyperband_multi_fidelity.py` — Outer Hyperband loop
  over brackets + inner successive halving. Toy: hyperparam =
  (lr, hidden); target near lr = 0.01, hidden = 64. R=81,
  η=3. Best-so-far loss found near the target region across
  5 brackets. Illustrates trade-off: bracket s=4 tries many
  configs at tiny budget, s=0 tries few at R budget.
- `r/hyperband_multi_fidelity.R` — `mlr3hyperband`,
  `mlr3tuning` + reticulate (R); Ray Tune HyperBandScheduler,
  Optuna HyperbandPruner, hpbandster, from-scratch (Python).

## When to use

- **Any partial-budget hyperparameter search** — training
  epochs, sub-sampled datasets, model size proxies.
- **Cheap-early, expensive-late** — losses at low budget are
  informative about high-budget rankings.
- **Parallel compute available** — Hyperband brackets are
  embarrassingly parallel.

## When NOT to use

- **Low-budget noise dominates** — if early-stopping decisions
  are near-random, Hyperband wastes budget. Consider BOHB or
  full-budget search.
- **Configs with huge start-up cost** — many low-budget runs
  add up.
- **Non-monotone budget-performance curves** — a config that
  starts poorly but ends best is discarded early.

## Assumptions & caveats

- **η (eta) tuning** — typical values 2-4; too high (η ≥ 5)
  prunes aggressively.
- **Bracket count** — S = ⌊log_η(R)⌋. Choose R (max budget)
  to fit compute budget.
- **Uniform config sampling** by default — combine with BO
  (BOHB) or TPE for smarter sampling.
- **Reporting** — best score across brackets is the standard
  metric; also report cumulative wall-clock savings vs
  random search.

## Related in this repo

- `successive-halving-asha` — the inner loop.
- `bohb-bayesian-hyperband` — Hyperband + TPE.
- `bayesian-optimization` — the non-multi-fidelity cousin.
- `tpe-tree-parzen-estimator` — SMBO sampler.

## Run

```
python techniques/hyperband-multi-fidelity/python/hyperband_multi_fidelity.py
Rscript techniques/hyperband-multi-fidelity/r/hyperband_multi_fidelity.R
```

**Refs:** Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. and Talwalkar, A. "Hyperband: A novel bandit-based approach to hyperparameter optimization." *J. Mach. Learn. Res.*, 18(185): 1-52, 2018.

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
