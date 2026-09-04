# Multivariable Model Building (Reference §39.2)

Harrell (2015), Steyerberg (2019 ch 11). Practical strategy for
building a **prediction** model. Harrell's message: stepwise
selection is harmful for prediction — inflated apparent performance,
unstable coefficients, invalidated p-values. The recommended
alternative is the **full-model strategy** with global shrinkage.

## Harrell's full-model strategy

1. **Pre-specify predictors** from clinical knowledge / literature —
   not data-driven univariable screening.
2. **Respect effective sample size**: `n / EPV ≥ ~15` for logistic
   or Cox models (`EPV` = events per predictor).
3. **Fit the full model** with all candidates simultaneously.
4. **Apply global shrinkage** (van Houwelingen-Le Cessie 1990) or
   ridge / LASSO to protect against optimism.
5. **Assess bootstrap-optimism-corrected performance** — never
   report apparent performance alone.

## van Houwelingen-Le Cessie heuristic shrinkage

```
s = (LR χ² − df) / LR χ²         (bounded [0, 1])
β_shrunk = s · β_full
```

Requires no cross-validation loop — heuristic that follows from
Efron's optimism formula.

## When to use

- **Every clinical prediction model** — this is the default modern
  workflow.

## When NOT to use

- **Inference on a single coefficient** — use a purposefully
  chosen DAG-based adjustment set, not the full-model + shrinkage
  strategy.
- **Very small samples** with prohibitively low EPV — collect more
  data first; a shrunken model on 5 EPV is still unreliable.

## Files

- `python/multivariable_model_building.py` — 3-way comparison at
  n=300, p=8 (2 true predictors + 6 noise): full logistic vs
  backward-AIC stepwise vs full + van Houwelingen shrinkage.
  Demo: **all three achieve ~0.85 CV AUC**; shrinkage factor
  **s = 0.942** shows modest but present optimism protection.
- `r/multivariable_model_building.R` — `rms::lrm`/`fastbw`/
  `pentrace`/`validate` (R); `sklearn.LogisticRegression` +
  custom (Python).

## Assumptions & caveats

- **EPV rule of thumb** — 10-15 events per candidate predictor is
  a guideline, not a law. Very rare outcomes need larger EPV.
- **Global shrinkage assumes homogeneity** of the shrinkage across
  coefficients; strong-signal + weak-signal mixtures may benefit
  from LASSO's differential shrinkage.
- **AIC-based stepwise** here is illustrative — Harrell recommends
  against it for prediction; use ridge / LASSO instead.
- **Do not report inference p-values** on the final model chosen
  by any data-driven procedure.

## Related in this repo

- `prediction-vs-inference` — why the goals differ.
- `bootstrap-optimism-correction` — the honest performance
  assessment.
- `penalized-clinical-prediction` — ridge / LASSO alternative.
- `nomograms`, `clinical-risk-scores` — deployment forms of the
  final model.

## Run

```
python techniques/multivariable-model-building/python/multivariable_model_building.py
Rscript techniques/multivariable-model-building/r/multivariable_model_building.R
```

**Refs:** Harrell, F.E. *Regression Modeling Strategies*, 2nd ed., Springer, 2015; Steyerberg, E.W. *Clinical Prediction Models*, 2nd ed., Springer, 2019 (ch 11); van Houwelingen, J.C. & Le Cessie, S. "Predictive value of statistical models." *Statistics in Medicine*, 1990.

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
