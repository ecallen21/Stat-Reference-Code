# Missing Indicator Method (Reference §41.12)

Groenwold et al. (2012), Knol et al. (2010). A pragmatic missing-
data strategy: **impute** the variable (mean, zero, or MICE) and
**add a binary indicator** for missingness. Include both in the
model.

## The trade-off

- **Prediction**: often helpful — missingness itself carries
  information when the observation process is informative
  (a lab test not ordered may reflect clinical judgement).
- **Inference / causal**: **biased unless MCAR** — Knol 2010
  demonstrates unpredictable bias in the target coefficient. Use
  multiple imputation instead.

## When to use

- **Prediction models** where missingness is informative and cannot
  be fixed at collection.
- **Retrospective claims / EHR analysis** where "test not ordered"
  is a common pattern.

## When NOT to use

- **Causal effect estimation** — MI (multiple imputation) is the
  right tool.
- **MCAR data** — the indicator is uninformative; complete-case
  analysis is unbiased.

## Files

- `python/missing_indicator_method.py` — mean impute + missing-
  indicator concatenation, then 5-fold CV RMSE with vs without
  indicators. Demo (n=800, p=3, X₀ MNAR with p(missing) tied to
  y, 51 % missingness): **CV RMSE 1.59 without → 1.26 with
  indicators** — indicators help prediction under MNAR.
- `r/missing_indicator_method.R` — `recipes::step_indicate_na`,
  `mice`, `hmisc` (R); `sklearn.impute.MissingIndicator` + custom
  (Python).

## Assumptions & caveats

- **MCAR** — indicator is uninformative; keep it out.
- **MAR** — MI is unbiased; the indicator method introduces
  unpredictable bias.
- **MNAR** — MI is also biased; indicators + domain knowledge are
  a pragmatic patch.
- **Multiple missing predictors** — one indicator per predictor;
  do not aggregate into a single "any missing" flag.
- **Interaction** — indicator × X interactions capture different
  slopes for missing vs observed cases.

## Related in this repo

- `data-drift-detection`, `covariate-shift-adaptation` — deployment
  cousins.
- `multiple-imputation` (if present) — proper inference under MAR.

## Run

```
python techniques/missing-indicator-method/python/missing_indicator_method.py
Rscript techniques/missing-indicator-method/r/missing_indicator_method.R
```

**Refs:** Groenwold, R.H.H., White, I.R., Donders, A.R.T., Carpenter, J.R., Altman, D.G., & Moons, K.G.M. "Missing covariate data in clinical research: when and when not to use the missing indicator method for analysis." *CMAJ*, 2012; Knol, M.J., Janssen, K.J.M., Donders, A.R.T. et al. "Unpredictable bias when using the missing indicator method." *Journal of Clinical Epidemiology*, 2010.

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
