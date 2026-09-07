# BART -- Bayesian Additive Regression Trees (Reference §5.16)

Chipman, George & McCulloch (2010). Sum-of-trees regression with
regularising priors:

    y = Σ_{k=1}^m g(x; T_k, M_k) + ε

with `m` trees, prior shrinkage keeping each tree a weak learner,
and MCMC over tree structure + leaf means (grow / prune / change /
swap Metropolis moves).

Posterior gives predictive credible intervals for `f(x)` and
variable-selection frequencies.

## Files

- `python/bart_bayesian_additive_regression_trees.py` — minimal
  BART surrogate: `m=40` stumps fitted round-robin on residuals
  from scratch. Demo (n=500, `y = sin(π x0) + 0.5 x1² + noise`):
  R² = 0.83, MSE = 0.20 — the machinery works; full BART's MCMC
  gives credible intervals + variable importance in `BART` /
  `dbarts` (R).
- `r/bart_bayesian_additive_regression_trees.R` — `BART`,
  `dbarts`, `bartMachine`, `BayesTree`, `stan4bart` (R);
  `pymc-bart`, `bartpy` (Python).

## When to use

- **Tabular prediction with UQ** — credible intervals for `f(x)`
  come for free.
- **Small-to-moderate n with nonlinear + interactions** — BART
  competes with gradient boosting on clean tabular data.
- **Causal inference** — BART is common in `bartCause`,
  Athey-Wager, and psASTI (variable-controlled) causal work.
- **Automatic variable selection** — inclusion frequencies flag
  useful predictors without pre-specified interactions.

## When NOT to use

- **Very large n** — MCMC is slow; XGBoost / LightGBM faster.
- **Deep / image / text inputs** — trees don't natively handle
  them.
- **Strong linearity** — regularised linear regression suffices.

## Assumptions & caveats

- **Prior tuning** (`sigma_mu`, tree-depth prior α, β,
  hyperprior on σ²) matters; defaults from Chipman 2010 usually
  fine.
- **Number of trees `m`** — 50-200 typical; more trees = smoother
  fit but slower.
- **MCMC convergence** — R̂ / trace plots on σ²; run multiple chains.
- **Variable importance** = posterior mean of the number of splits
  involving a variable; not causal.

## Related in this repo

- `random-forest`, `gradient-boosting`,
  `explainable-boosting-machine`, `decision-tree` — tree
  siblings.
- `bayesian-hierarchical-models`, `gaussian-process-regression`,
  `probabilistic-pca` — Bayesian regression cousins.
- `causal-forest`, `dml-double-ml` — causal-tree neighbours.

## Run

```
python techniques/bart-bayesian-additive-regression-trees/python/bart_bayesian_additive_regression_trees.py
Rscript techniques/bart-bayesian-additive-regression-trees/r/bart_bayesian_additive_regression_trees.R
```

**Refs:** Chipman, H.A., George, E.I. & McCulloch, R.E. "BART: Bayesian additive regression trees." *Annals of Applied Statistics*, 4(1): 266-298, 2010.

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
