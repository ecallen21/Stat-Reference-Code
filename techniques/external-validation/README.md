# External Validation (Reference §39.5)

Steyerberg & Harrell (2016), Collins et al. (2012). Fit the model on
a **development** cohort and evaluate it on an **independent**
validation cohort. Three flavours by design:

- **Temporal** — different time window, same institution.
- **Geographic** — different sites / regions.
- **Domain** — different case-mix (e.g., primary vs tertiary care).

## What to report

**Discrimination** (`AUC`) plus **calibration**:

- **CITL** (calibration-in-the-large) = `mean(y) − mean(p̂)`;
  ideal 0.
- **Calibration slope** = slope of `y ~ logit(p̂)`; ideal 1.
  `< 1` → over-fit (predictions too extreme); `> 1` → under-fit.
- **Brier score** = `mean((p̂ − y)²)`; lower better.

## Recalibration

If discrimination survives but calibration is off, refit intercept
alone ("recalibration in the large") or intercept + slope
("logistic recalibration") — see `model-recalibration`.

## When to use

- **Every clinical prediction model** intended for use beyond its
  development sample.
- **Temporal drift monitoring** — refit annually to detect calibration
  drift.

## When NOT to use

- **Development sample too small for a valid model** — external
  validation of a broken model is misleading; fix the model first.
- **Population identical to development** — that's internal
  validation.

## Files

- `python/external_validation.py` — logistic fit on development
  cohort, evaluate on shifted-covariate external cohort. Demo
  (n_dev=800, n_ext=500, covariates shifted + coefficients 15 %
  smaller): **development AUC 0.765, external AUC 0.768**
  (discrimination preserved); slope drifts **1.00 → 0.875**
  (external over-fit warning); CITL −0.026.
- `r/external_validation.R` — `rms::val.prob`, `riskRegression::
  Score`, `predtools::calibration_plot` (R); `sklearn.metrics` +
  custom (Python).

## Assumptions & caveats

- **Independent cohort** — sharing patients or sites biases the
  validation toward apparent performance.
- **Same outcome definition** — outcome-drift kills any
  interpretation of calibration.
- **Same predictors and coding** — dummy definitions, units, and
  missingness handling must match the development pipeline.
- **Case-mix vs coefficient shift** — CITL flags marginal mis-
  calibration; slope flags predictor-effect shift. Both matter.
- **Report all three cohorts**: development, internal-validation
  (bootstrap-optimism-corrected), external.

## Related in this repo

- `bootstrap-optimism-correction` — internal-validity companion.
- `model-recalibration` — the response to calibration drift.
- `calibration-plots` — visual diagnostic.
- `iecv-multisite` — leave-one-site-out design.

## Run

```
python techniques/external-validation/python/external_validation.py
Rscript techniques/external-validation/r/external_validation.R
```

**Refs:** Steyerberg, E.W. & Harrell, F.E. "Prediction models need appropriate internal, internal-external, and external validation." *Journal of Clinical Epidemiology*, 2016; Collins, G.S., de Groot, J.A., Dutton, S. et al. "External validation of multivariable prediction models: a systematic review." *BMJ*, 2012.

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
