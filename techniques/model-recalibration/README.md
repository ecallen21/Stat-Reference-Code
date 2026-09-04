# Model Recalibration (Reference §39.6)

Steyerberg (2019 ch 20), Janssen et al. (2008). When a prediction
model's calibration drifts on a new cohort but discrimination
survives, **recalibrate** rather than refit from scratch.

## Three levels of update

- **A. Recalibration in the large** — intercept-only shift
  `new_lp = old_lp + α`. Fixes systemic over- or under-prediction.
- **B. Logistic recalibration** — intercept + slope update
  `new_lp = α + β · old_lp`. Also corrects predictor-effect drift
  in aggregate.
- **C. Model revision** — refit individual coefficients on the new
  data. Loses the original identity → needs its own validation.

Prefer the **least invasive** fix that resolves the drift.

## Discrimination invariance

A and B are monotone transformations of the linear predictor →
AUC is unchanged; only calibration and probability magnitudes shift.

## When to use

- **Temporal drift** on annual monitoring.
- **Case-mix differences** between institutions.
- **Deploying an off-the-shelf model** in a new health system.

## When NOT to use

- **Discrimination has collapsed** — recalibration cannot rescue a
  model whose predictors have lost their signal.
- **Both cohorts are the same** — recalibration on the development
  cohort is a no-op / a tautology.

## Files

- `python/model_recalibration.py` — intercept-only + logistic
  recalibration on an externally-shifted cohort. Demo (dev n=800,
  ext n=500, external coefficients 15 % smaller, baseline risk
  shifted): raw external CITL = **−0.098**, Brier 0.205; intercept
  fix (α = −0.41) → CITL **−0.019**, Brier 0.194; logistic fix
  (α = −0.53, β = 0.79) → CITL **0.000**, Brier 0.192.
- `r/model_recalibration.R` — `rms::val.prob`,
  `predtools::recalibrate`, `pmcalibration` (R);
  `sklearn.calibration.CalibratedClassifierCV` + custom (Python).

## Assumptions & caveats

- **Recalibrate on independent data** — same cohort you validate on
  ≠ same cohort you refit on; split or use cross-fitting.
- **Update ≠ development** — a recalibrated model still needs
  temporal re-validation on the next cohort.
- **Report the update** — publish α, β, and CITL both before and
  after recalibration.
- **Logistic vs isotonic** — logistic recalibration assumes linear
  drift; isotonic (Platt / isotonic-regression) is non-parametric
  but greedier with data.

## Related in this repo

- `external-validation` — the diagnostic that motivates
  recalibration.
- `calibration-plots`, `discrimination-calibration` — verify the
  fix worked.
- `calibration-scaling` — Platt / isotonic ML-side cousins.

## Run

```
python techniques/model-recalibration/python/model_recalibration.py
Rscript techniques/model-recalibration/r/model_recalibration.R
```

**Refs:** Steyerberg, E.W. *Clinical Prediction Models*, 2nd ed., Springer, 2019 (ch 20); Janssen, K.J.M., Moons, K.G.M., Kalkman, C.J., Grobbee, D.E., & Steyerberg, E.W. "Updating methods improved the performance of a clinical prediction model in new patients." *Journal of Clinical Epidemiology*, 2008.

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
