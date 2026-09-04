# Penalised Regression for Clinical Prediction (Reference §39.9)

Steyerberg (2019 ch 12), van Houwelingen & Le Cessie (1990). When
events per predictor (EPV) is low, unpenalised MLE overfits.
Regularisation shrinks or selects coefficients to protect the model.

## Choices

- **Ridge (L2)** — shrinks all coefficients toward zero uniformly.
  Choose `λ` by CV or effective-df target (`rms::pentrace`).
- **LASSO (L1)** — shrinks and **selects** (some coefficients
  exactly zero). Useful when a sparse truth is plausible.
- **Elastic net** — `α · L1 + (1 − α) · L2`. Handles correlated
  predictors better than pure LASSO.
- **Global shrinkage (van Houwelingen-Le Cessie)** — analytic
  one-shot shrinkage factor `s = (χ² − df) / χ²` applied to all
  coefficients (see `multivariable-model-building`).

## When to use

- **Low EPV** — `< 15 events per predictor` is the classical
  danger zone.
- **Correlated predictors** — ridge / elastic net stabilises the
  fit.
- **Sparse truth** — LASSO both prevents overfitting and produces
  a shorter, deployable model.

## When NOT to use

- **Inference target** — regularised confidence intervals lose
  frequentist coverage without special methods (debiased-lasso,
  see `debiased-lasso`).
- **Very small `n`** where even penalisation can't stabilise the
  fit — collect more data first.

## Files

- `python/penalized_clinical_prediction.py` — 3-way comparison
  (unpenalised, ridge-CV, lasso-CV) at n=200, p=15, EPV=5.9 with
  4 true + 11 noise predictors. Demo: CV AUC **unpenalised 0.617,
  ridge 0.617, lasso 0.644** (lasso picks 1 predictor); max |β̂|
  shrinks from 0.68 (unpenalised) → 0.27 (ridge).
- `r/penalized_clinical_prediction.R` — `glmnet::cv.glmnet`,
  `rms::pentrace`, `caret` (R);
  `sklearn.linear_model.LogisticRegressionCV`/`LassoCV`/
  `ElasticNetCV` (Python).

## Assumptions & caveats

- **Standardise predictors** before penalisation — otherwise
  scale differences distort the penalty.
- **Tune `λ` inside the CV loop** — external CV of CV-selected `λ`
  is the honest workflow; a single-pass CV inflates apparent
  performance.
- **LASSO instability** with correlated features — coefficients
  flip between correlated predictors across bootstrap resamples.
  Use elastic net or stability selection.
- **Report the effective sample size / penalty** so users can
  reproduce the fit.

## Related in this repo

- `multivariable-model-building` — full-model + van Houwelingen
  shrinkage.
- `debiased-lasso` — inference after LASSO.
- `adaptive-lasso`, `group-lasso`, `stability-selection` — modern
  variants.
- `bootstrap-optimism-correction` — internal-validity for the
  regularised model.

## Run

```
python techniques/penalized-clinical-prediction/python/penalized_clinical_prediction.py
Rscript techniques/penalized-clinical-prediction/r/penalized_clinical_prediction.R
```

**Refs:** Steyerberg, E.W. *Clinical Prediction Models*, 2nd ed., Springer, 2019 (ch 12); van Houwelingen, J.C. & Le Cessie, S. "Predictive value of statistical models." *Statistics in Medicine*, 1990; Tibshirani, R. "Regression shrinkage and selection via the lasso." *JRSS-B*, 1996.

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
