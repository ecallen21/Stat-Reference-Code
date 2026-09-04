# Prediction vs Inference Framework (Reference §39.1)

Shmueli (2010), Steyerberg (2019). Prediction and inference share
model machinery but have **fundamentally different goals** — and so
different variable-selection rules, regularisation strategies, and
performance metrics.

| Goal | Question | Selection | Reporting |
|---|---|---|---|
| **Inference** | What's the effect of X on Y adjusting for confounders? | Change-in-estimate / DAG-informed | β̂, SE, CI, hypothesis tests |
| **Prediction** | What's my best guess of Y for a new patient? | Cross-validated performance | Discrimination + calibration on held-out data |

## Rules of thumb

- **Include confounders for inference** whether or not they improve
  predictive power — they debias the target coefficient.
- **Include predictors for prediction** only if they improve out-of-
  sample performance — a "significant" predictor with near-zero
  effect can hurt cross-validated MSE by adding variance.
- **Never use stepwise for a prediction model** (Harrell 2015 ch 4);
  it inflates apparent performance and yields unstable coefficients.
- **Regularise (ridge / LASSO / global shrinkage)** for prediction;
  inference intervals lose their frequentist coverage under
  regularisation.

## Files

- `python/prediction_vs_inference.py` — same simulated `(x, z, y)`
  data, two decisions: change-in-estimate for `β_x` (inference)
  vs 10-fold CV MSE (prediction). Demo: **β_x alone = 2.96 vs
  adjusted = 0.98** (67 % change → include `z` for inference); CV
  MSE **2.48 → 0.88** (also include `z` for prediction). Contrast:
  a noise predictor `u` correlated with `x` but unrelated to `y`
  changes `β_x` by 0.1 % and CV MSE by 0.3 % — both frameworks
  correctly drop it.
- `r/prediction_vs_inference.R` — `rms` (inference-first),
  `caret`/`tidymodels` (prediction-first); Python `statsmodels` +
  `sklearn`.

## When to use

- **Every modelling project** should explicitly state which frame
  it is in before any variable-selection step.

## When NOT to use

- **Dual-purpose reports** are hazardous: a single model chosen for
  prediction should not carry inference-style p-values on
  individual coefficients.

## Related in this repo

- `multivariable-model-building` — Harrell's full-model strategy.
- `bootstrap-optimism-correction`, `external-validation` —
  prediction-side workflow.
- `dags-and-confounding` (if present) — inference-side variable
  selection.

## Run

```
python techniques/prediction-vs-inference/python/prediction_vs_inference.py
Rscript techniques/prediction-vs-inference/r/prediction_vs_inference.R
```

**Refs:** Shmueli, G. "To Explain or to Predict?" *Statistical Science*, 2010; Steyerberg, E.W. *Clinical Prediction Models*, 2nd ed., Springer, 2019; Harrell, F.E. *Regression Modeling Strategies*, 2nd ed., Springer, 2015.

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
