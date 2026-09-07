# Random Survival Forest (Reference §11.24)

Ishwaran, Kogalur, Blackstone & Lauer (2008). Extension of Breiman's
random forests to right-censored survival data.

## Recipe

- **Bagging.** Each tree fits on a bootstrap sample of the data
  (~63.2% in-bag, ~36.8% OOB).
- **mtry.** At each split, a random subset of covariates is considered.
- **Split rule.** Two-sample **log-rank** statistic, maximised over
  candidate cut points at each node.
- **Terminal-node hazard.** Nelson-Aalen cumulative hazard
  `H(t) = Σ dⱼ / Rⱼ` over the events within the leaf.
- **Ensemble.** Predicted CHF for a new subject = average of the leaf
  CHFs across trees.

## Files

- `python/random_survival_forest.py` — vectorised log-rank split +
  quantile candidate cuts + Nelson-Aalen leaves + ensemble CHF from
  scratch. Demo (n=250, p=5, censoring 24%, true effects on x0, x1):
  Harrell C = 0.742 vs Cox baseline ≈ 0.65-0.72.
- `r/random_survival_forest.R` — `randomForestSRC::rfsrc`,
  `ranger::ranger(splitrule="logrank")` (R);
  `sksurv.ensemble.RandomSurvivalForest`, `pysurvival` (Python).

## When to use

- **Nonlinear / interaction-heavy hazards** — Cox misses; a forest
  captures the shape without a specified functional form.
- **High-dimensional predictors** — RSF handles wide X with variable
  importance; mtry regularises.
- **Competing risks** — `randomForestSRC` handles them natively; use
  event-specific CHFs.

## When NOT to use

- **Small n / few events** — the log-rank split rule is unstable when
  each leaf has < ~10 events.
- **You need coefficients** — RSF is a prediction tool; for effect
  estimation use Cox / AFT.
- **Interpretable subgroups** — RSF variable importance is helpful,
  but subgroup exposition needs a stump or a rule extractor.

## Assumptions & caveats

- **Non-informative censoring** — same as Cox / KM.
- **Terminal-node size** — too-small leaves fit noise; RSF's
  `nsplit`, `mtry`, and minimum-events-per-leaf parameters matter.
- **Prediction extrapolation** — CHF beyond observed times is
  effectively 0 hazard change; report only within the follow-up
  window.
- **C-index caveats** — Harrell C ignores censoring distribution;
  Uno's C or Brier score for time-dependent evaluation.

## Related in this repo

- `cox-ph`, `cox-time-varying`, `parametric-survival` — parametric /
  semi-parametric alternatives.
- `random-forest`, `gradient-boosting` — non-survival tree ensembles.
- `harrell-c-index` — the discrimination metric used above.

## Run

```
python techniques/random-survival-forest/python/random_survival_forest.py
Rscript techniques/random-survival-forest/r/random_survival_forest.R
```

**Refs:** Ishwaran, H., Kogalur, U.B., Blackstone, E.H. & Lauer, M.S. "Random survival forests." *Annals of Applied Statistics*, 2(3): 841-860, 2008.

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
