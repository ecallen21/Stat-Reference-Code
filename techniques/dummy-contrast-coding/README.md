# Dummy / Effect / Contrast Coding (Reference §41.6)

Cohen-Cohen-West-Aiken (2003), Davis (2010). Different ways to
represent a `K`-level categorical predictor in a regression design
matrix. Each coding gives the **same fit and predictions**; only the
**meaning of the coefficients** changes.

| Coding | β interpretation |
|---|---|
| **Reference / dummy** | Each β = mean vs the reference level |
| **Effect / deviation** | Each β = deviation from the grand mean |
| **Helmert** | β_j = level j vs mean of levels 1..j−1 |
| **Polynomial** | Orthogonal linear / quadratic / … for ordered factors |

## When to use

- **Reference** — clinical comparisons vs a control group.
- **Effect** — balanced designs where each level's departure from
  overall matters.
- **Helmert** — nested comparisons (e.g., dose escalation).
- **Polynomial** — ordered factors where a smooth trend is
  substantive.

## When NOT to use

- **Free-form categorical** with no natural reference — pick effect
  or drop the intercept.
- **Reporting audience unfamiliar** — pick the coding whose
  interpretation matches the audience's question.

## Files

- `python/dummy_contrast_coding.py` — dummy, effect, and Helmert
  design-matrix construction + regression on the same y showing
  fits agree while coefficient meanings differ.
- `r/dummy_contrast_coding.R` — `stats::contr.treatment` /
  `contr.sum` / `contr.helmert` / `contr.poly` / `contr.SAS`,
  `stats::model.matrix`, `fastDummies`, `recipes::step_dummy` (R);
  `patsy.C()`, custom (Python).

## Assumptions & caveats

- **Intercept meaning** changes with the coding — reference =
  reference-level mean; effect = grand mean.
- **Interactions** — dummy × dummy interaction coefficients are
  simple-effect contrasts at the reference; effect × effect are
  centred effects. Pick the coding to match the effect you want to
  report.
- **Collinearity** — never include both the drop-one dummies and
  their inverse.
- **Missing categories at prediction** — new levels break dummy
  coding; hashing or robust label encoders handle out-of-vocabulary
  categories.

## Related in this repo

- `standardized-coefficients` — standardising categorical predictors.
- `feature-hashing`, `target-encoding` — alternatives for high-
  cardinality categoricals.
- `standardization-scaling` — companion numeric preprocessing.

## Run

```
python techniques/dummy-contrast-coding/python/dummy_contrast_coding.py
Rscript techniques/dummy-contrast-coding/r/dummy_contrast_coding.R
```

**Refs:** Cohen, J., Cohen, P., West, S.G., & Aiken, L.S. *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences*, 3rd ed., Routledge, 2003 (ch 8); Davis, M.J. "Contrast coding in multiple regression analysis: strengths, weaknesses, and utility of popular coding structures." *Journal of Data Science*, 2010.

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
