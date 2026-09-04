# Bootstrap Optimism Correction (Reference §39.4, §39.16)

Efron (1983), Harrell (2015 ch 5), Steyerberg (2019 ch 5). Fitting
and testing on the same data yields **apparent** performance that is
too optimistic. The Efron-Harrell bootstrap correction estimates the
overfitting gap directly and subtracts it from the apparent metric.

## Algorithm

For a metric `M(model, data)`:

```
apparent = M(fit(D), D)
for b in 1..B:
    D_b  ← bootstrap resample of D (n from n with replacement)
    M_b  ← fit(D_b)
    diff = M(M_b, D_b) − M(M_b, D)      # train − test on same fitted model
optimism  = mean(diff)
corrected = apparent − optimism
```

Works for any metric (AUC, Brier, calibration slope, R²).

## When to use

- **Every clinical prediction model** report should carry
  bootstrap-optimism-corrected discrimination and calibration —
  not just apparent numbers.
- **Small n / low EPV** — cross-validation has high variance; the
  bootstrap gives a stabler estimate of internal validity.

## When NOT to use

- **Large held-out test set already available** — direct external
  validation is more honest.
- **Model class does not converge on bootstrap resamples**
  (e.g. some fully-connected NN training) — use CV instead.

## Files

- `python/bootstrap_optimism_correction.py` — generic bootstrap
  optimism loop for AUC and Brier score. Demo (n=120, p=12,
  EPV=3.8, moderate signal): **apparent AUC 0.752 → optimism-
  corrected 0.651** (optimism +0.101); Brier apparent 0.198 →
  corrected 0.247 (optimism −0.048; lower Brier = better).
- `r/bootstrap_optimism_correction.R` — `rms::validate` (canonical
  R workflow), `rms::calibrate`, `caret::train`, `boot` (R);
  `sklearn.utils.resample` + custom (Python).

## Assumptions & caveats

- **B = 200-500** is standard; 500 for publication.
- **Model refitting speed** — heavy models need parallelisation.
- **Same fit protocol** — the bootstrap loop must include every
  data-driven step (imputation, selection, tuning) or optimism is
  under-estimated.
- **Metric monotonicity** — for metrics where lower is better
  (Brier), "optimism" is negative and the correction subtracts a
  negative → the corrected value is worse (larger).
- **Distributional shift** — bootstrap optimism captures overfit
  to the training distribution, not shift to a different
  population. Pair with external validation.

## Related in this repo

- `multivariable-model-building` — the model whose optimism you
  correct.
- `external-validation` — the transportability question.
- `iecv-multisite` — a multi-site cousin.
- `discrimination-calibration`, `calibration-plots` — the metrics
  you correct.

## Run

```
python techniques/bootstrap-optimism-correction/python/bootstrap_optimism_correction.py
Rscript techniques/bootstrap-optimism-correction/r/bootstrap_optimism_correction.R
```

**Refs:** Efron, B. "Estimating the error rate of a prediction rule: improvement on cross-validation." *JASA*, 1983; Harrell, F.E. *Regression Modeling Strategies*, 2nd ed., Springer, 2015 (ch 5); Steyerberg, E.W. *Clinical Prediction Models*, 2nd ed., Springer, 2019 (ch 5, 17).

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
