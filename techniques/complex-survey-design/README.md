# Complex Survey Design Analysis (Reference §27.7)

Kish (1965); Cochran (1977); Lumley (2010). Design-based inference
from stratified, clustered, unequal-probability surveys.

## Building blocks

- **Sampling weight** `w_i = 1 / π_i` — inverse inclusion probability.
- **Horvitz-Thompson estimator** of the total: `T̂ = Σ w_i · y_i`.
- **Weighted mean** (Hájek): `ȳ = T̂ / N̂` with `N̂ = Σ w_i`.
- **Variance**:
  - Linearisation (Taylor) — analytical.
  - Balanced Repeated Replicates (BRR / Fay).
  - Delete-1 PSU jackknife (Rao-Wu).
  - Rescaled bootstrap (Rao-Wu-Yue).

## Files

- `python/complex_survey_design.py` — HT weighted mean + stratified
  linearisation variance + delete-1-PSU jackknife (Rao-Wu) from
  scratch. Demo (4 strata × 5 PSUs × 20 SSUs, unequal weights):
  naive mean 2.96 vs HT weighted 3.94 (weights shift the estimate);
  linearisation SE 0.054 vs jackknife SE 0.062.
- `r/complex_survey_design.R` — `survey::svydesign / svymean /
  svyglm`, `survey::as.svrepdesign`, `srvyr`, `sampling` (R);
  `samplics`, `statsmodels.stats.weightstats` (Python).

## When to use

- **Any survey with non-SRS design** — stratification, clustering,
  unequal probabilities, calibration weights.
- **National health / social surveys** — NHANES, BRFSS, NHIS,
  MEPS, HRS.
- **Sub-population estimates** — need domain-appropriate SEs.
- **Weighted regression** — GLM with survey weights (`svyglm`).

## When NOT to use

- **iid convenience samples** — no weights, no clustering; OLS is
  fine.
- **Model-based inference** — random-effects on cluster ID is a
  different (and often complementary) framework.
- **Non-probability samples** — MRP (multilevel regression +
  post-stratification) is a better fit than pure design-based.

## Assumptions & caveats

- **Inclusion probabilities known** — `π_i` (or a proxy, calibration
  weights) required for HT.
- **Non-response adjustment** — weights typically absorb it via
  post-stratification / raking; document how.
- **Domain analysis** — a sub-population mean uses full-design
  variance, not the sub-population SRS variance.
- **Software agreement** — `survey::svy*` (R) and `samplics` /
  Stata `svy` return equivalent SEs when the design object is
  identical.

## Related in this repo

- `two-stage-cluster-sampling` — a specific multi-stage design.
- `fay-herriot-small-area`, `poisson-gamma-empirical-bayes` —
  model-based small-area alternatives.
- `nonparametric-bootstrap`, `jackknife` — variance resampling
  cousins.
- `iptw` — sample-weighting for causal identification (different
  goal but same mechanics).

## Run

```
python techniques/complex-survey-design/python/complex_survey_design.py
Rscript techniques/complex-survey-design/r/complex_survey_design.R
```

**Refs:** Kish, L. *Survey Sampling*, Wiley, 1965; Cochran, W.G. *Sampling Techniques*, 3rd ed., Wiley, 1977; Lumley, T. *Complex Surveys: A Guide to Analysis using R*, Wiley, 2010.

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
