# Tree-structured Parzen Estimator (Reference §47.278)

Bergstra, Bardenet, Bengio & Kégl (2011). SMBO variant using
density-ratio Bayesian optimisation. Partition observed
`(x, loss)` into GOOD (bottom γ quantile) and BAD; model
kernel densities; propose the next `x` to maximise:

```
EI(x) ∝ p(x | good) / p(x | bad)
```

Handles conditional / mixed-type spaces via a tree structure.
Default optimiser in hyperopt / Optuna.

## Files

- `python/tpe_tree_parzen_estimator.py` — 1-D TPE with kernel
  Parzen densities. Toy: 30-step search on a wiggly function
  with min near x=0.7. After ~15 steps, TPE concentrates
  proposals near x=0.83; final best x≈0.83 loss≈−0.06 (true
  best region 0.7-0.9 due to the sine ripple).
- `r/tpe_tree_parzen_estimator.R` — `mlr3mbo`, reticulate +
  hyperopt / Optuna (R); hyperopt.tpe.suggest, Optuna
  TPESampler, from-scratch (Python).

## When to use

- **Hyperparameter search** with mixed continuous / categorical
  / conditional HPs — TPE's tree structure handles them
  natively.
- **Cheap evaluations (≤ 1000)** — TPE's overhead is modest.
- **Prior knowledge as warm start** — feed hand-picked
  configs as history.

## When NOT to use

- **Extreme-dimensional HPs (> 30)** — kernel densities
  suffer; use CMA-ES / random search + bandits.
- **Very expensive evaluations (hours each)** — Gaussian-
  process BO with EI is often sample-more-efficient.
- **Highly-correlated HPs** — TPE assumes tree-structured
  independence; correlated HPs may need SMAC or GP-BO.

## Assumptions & caveats

- **γ (quantile split)** — 0.15 - 0.25 is standard; too small
  = no good density, too large = too broad.
- **Bandwidth** — Parzen bandwidth affects convergence; use
  cross-validated or heuristic bandwidth.
- **Multi-objective** — Vanilla TPE is single-objective;
  extensions exist (MOTPE).
- **Concurrent trials** — hyperopt supports parallel via
  MongoDB; Optuna via RDB storage.

## Related in this repo

- `bayesian-optimization` — the GP-based cousin.
- `hyperband-multi-fidelity` — for cheap-early-scoring
  problems.
- `bohb-bayesian-hyperband` — TPE + Hyperband combination.
- `active-learning-query-strategies` — related uncertainty
  sampling.

## Run

```
python techniques/tpe-tree-parzen-estimator/python/tpe_tree_parzen_estimator.py
Rscript techniques/tpe-tree-parzen-estimator/r/tpe_tree_parzen_estimator.R
```

**Refs:** Bergstra, J., Bardenet, R., Bengio, Y. and Kégl, B. "Algorithms for hyper-parameter optimization." In *NeurIPS 24*, 2011.

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
