# Proper Scoring Rules — CRPS (Reference §47.255)

Gneiting & Raftery (2007). The CONTINUOUS RANKED PROBABILITY
SCORE assesses a predictive CDF `F` against an observed value
`y`:

```
CRPS(F, y) = ∫ (F(x) - 1{x ≥ y})² dx
```

For an EMPIRICAL forecast ensemble `{x_1, ..., x_n}`:

```
CRPS = (1/n) Σ |x_i - y| - (1/(2 n²)) Σ_{i,j} |x_i - x_j|
```

Strictly proper: minimised only by the true distribution.
Standard skill score for weather, finance, epidemic forecasts.

## Files

- `python/proper_scoring_rules_crps.py` — Gaussian closed-form
  + ensemble estimator with sorted-trick O(n log n). Demo: 500
  observations from N(0, 1). Perfect N(0, 1) forecast: CRPS
  ≈ 0.234 (min); biased N(0.5, 1): CRPS ≈ 0.343;
  over-dispersed N(0, 2): CRPS ≈ 0.659; under-dispersed
  N(0, 0.5): CRPS ≈ 0.617. Both bias and mis-scale pay a
  penalty — the propriety guarantee.
- `r/proper_scoring_rules_crps.R` — `scoringRules::crps`,
  `scoringutils`, `verification`, `ensembleBMA` (R);
  `properscoring.crps_ensemble` / `crps_gaussian`,
  from-scratch (Python).

## When to use

- **Probabilistic forecasts** — weather (Gneiting's home
  domain), epidemic curves, price distributions.
- **Ensemble evaluation** — compare N-member ensembles across
  models on the same units as the observation.
- **Model comparison** — CRPS decomposes into reliability +
  resolution + uncertainty (Hersbach 2000).

## When NOT to use

- **Point forecasts only** — use MAE / RMSE instead.
- **Discrete outcomes** — use log score, Brier for binary,
  ranked probability score for ordinal.
- **Extreme-tail focus** — CRPS is L2-flavoured and can
  under-emphasise tails; use quantile / threshold-weighted
  CRPS.

## Assumptions & caveats

- **Same units as observation** — CRPS = |y − ŷ| for a point
  forecast, so directly interpretable.
- **Ensemble size** — the O(n²) formula reduces to O(n log n)
  after sorting; still expensive for huge ensembles, but
  100–1000 members are fine.
- **Skill scores** — usually reported as CRPSS = 1 −
  CRPS_model / CRPS_ref (climatology as reference).
- **Multivariate outcomes** — extend to Energy Score (Gneiting
  & Raftery 2007) or Variogram Score.

## Related in this repo

- `discrimination-calibration` — Brier decomposition cousin.
- `calibration-plots` — visual diagnostics.
- `model-recalibration` — how to fix mis-calibrated
  probabilities.
- `cross-entropy-log-loss`, `information-criteria` — related
  proper scoring rules.
- `prediction-intervals` — the interval flavour.

## Run

```
python techniques/proper-scoring-rules-crps/python/proper_scoring_rules_crps.py
Rscript techniques/proper-scoring-rules-crps/r/proper_scoring_rules_crps.R
```

**Refs:** Gneiting, T. and Raftery, A.E. "Strictly Proper Scoring Rules, Prediction, and Estimation." *J. Amer. Statist. Assoc.*, 102(477): 359-378, 2007.

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
