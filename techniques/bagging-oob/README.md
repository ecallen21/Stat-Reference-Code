# Bagging + Out-Of-Bag Error (Reference §47.58)

Breiman (1996). Bootstrap-aggregate B independent learners on
IID bootstrap resamples; average predictions:
`f_bag(x) = (1/B) Σ_b f_b(x)`. Reduces variance without changing
bias. **Out-Of-Bag (OOB)** estimate: for each training point i,
average predictions from bags that did NOT include i — free
CV-like estimate at zero extra cost.

    Prob(i missed by a bag) = (1 − 1/n)ⁿ → 1/e ≈ 0.368.

## Files

- `python/bagging_oob.py` — from-scratch bagging with
  `DecisionTreeRegressor` base learners + OOB R² accounting.
  Demo (n=500, p=10, mixed nonlinear signal):
  - Single deep tree R² (5-fold CV) = 0.493
  - Bagging B=10:  OOB R² = 0.582, coverage 0.988
  - Bagging B=50:  OOB R² = 0.704, coverage 1.000
  - Bagging B=200: OOB R² = 0.707, coverage 1.000.
- `r/bagging_oob.R` — `ipred::bagging`,
  `randomForest::randomForest` (implicit bagger) (R);
  `sklearn.BaggingRegressor`, from-scratch (Python).

## When to use

- **High-variance base learners** — deep trees, MARS, neural
  nets on small data.
- **When CV budget is tight** — OOB gives free validation.
- **Feature-importance stability** — permutation importance from
  bagged ensembles.

## When NOT to use

- **Highly biased weak learners** — averaging preserves bias.
- **Correlated resamples** — bagging benefit diminishes; use
  random subspace or Random Forest.
- **Extreme class imbalance** — some bags may miss the minority
  class entirely; use balanced bagging / SMOTE.

## Assumptions & caveats

- **Independent bootstraps** — the standard scheme; block bootstrap
  for time series.
- **OOB is slightly pessimistic** — trains on ~63.2 % of data per
  bag; a k-fold cousin.
- **Choice of B** — OOB stabilises around B = 100–500 for trees;
  more helps little.
- **Not a bias reducer** — pair with boosting when bias dominates.

## Related in this repo

- `random-forest`, `random-survival-forest`, `causal-forest` —
  bagging + feature-subspace descendants.
- `bootstrap-optimism-correction`, `nonparametric-bootstrap`,
  `block-bootstrap`, `bca-bootstrap`, `wild-bootstrap` — resampling
  cousins.
- `gradient-boosting`, `bart-bayesian-additive-regression-trees` —
  ensemble contrasts (boosting vs bagging).
- `deep-ensembles`, `swag`, `mc-dropout` — modern uncertainty via
  ensembling.

## Run

```
python techniques/bagging-oob/python/bagging_oob.py
Rscript techniques/bagging-oob/r/bagging_oob.R
```

**Refs:** Breiman, L. "Bagging predictors." *Machine Learning* 24(2): 123-140, 1996; Breiman, L. "Random forests." *Machine Learning* 45(1): 5-32, 2001.

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
