# Nested Case-Control (NCC) Study (Reference §15.42)

Thomas (1977); Langholz & Goldstein (1996). Within a defined cohort,
select `K` controls per case matched on **risk-set membership**
(alive and uncensored at the case's event time). Conditional logistic
regression on the matched sets approximates the Cox partial-
likelihood hazard ratio.

## Advantages

- Cheap when **expensive covariate assessment** (biomarker,
  imaging, chart review) is needed only on cases + selected
  controls.
- Preserves the temporal / risk-set structure of survival data.
- Works with time-varying covariates by conditioning on the
  case time.

## Files

- `python/nested_case_control.py` — risk-set matcher + conditional
  logistic MLE from scratch. Demo (cohort n=4000, HR=3, event rate
  32%): recovers HR = 3.21 (1:1), 3.18 (1:2), 3.27 (1:4) across
  1260 matched sets.
- `r/nested_case_control.R` — `Epi::ccwc`, `survival::clogit`,
  `multipleNCC`, `mstate` (R); `statsmodels.discrete.
  conditional_models.ConditionalLogit`, `lifelines.CoxPHFitter`
  (full-cohort baseline) (Python).

## When to use

- **Expensive covariates** — biobank samples, imaging,
  genotyping — only need to collect on cases + controls.
- **Rare exposures / rare outcomes** — full-cohort Cox would be
  wasteful; NCC gives ≥ 90% relative efficiency with K=4-5.
- **Time-varying / historical exposures** — the risk-set match keeps
  time alignment.
- **Case-cohort alternative** — when a fixed sub-cohort is more
  natural, use `case-cohort` (a different design; see relevance
  note below).

## When NOT to use

- **Cheap covariates on everyone** — do the full-cohort Cox; more
  efficient.
- **Absolute-risk estimation** — NCC estimates hazard ratios, not
  absolute risks; use Cox with Breslow baseline or Aalen-Johansen.
- **Case-cohort target** — reuses the same sub-cohort across
  outcomes; NCC controls are outcome-specific.

## Assumptions & caveats

- **Matching on all risk-set defining variables** — typically age
  and calendar time; missing them biases HR.
- **Number of controls K** — efficiency plateaus by K=4-5;
  additional controls give diminishing returns.
- **Sampled controls can become cases later** — that's fine;
  Langholz-Goldstein show the partial likelihood is unaffected.
- **Weighted case-cohort** analyses give absolute-risk estimates
  but need different software (`survival::cch`).

## Related in this repo

- `cox-ph`, `cox-time-varying` — full-cohort survival cousins.
- `case-crossover` — self-controlled cousin using time-within-
  person.
- `propensity-score-matching` — matching by covariate similarity
  rather than by risk set.
- `sccs-self-controlled`, `disproportionality-signal-detection` —
  pharmacoepi cousins.

## Run

```
python techniques/nested-case-control/python/nested_case_control.py
Rscript techniques/nested-case-control/r/nested_case_control.R
```

**Refs:** Thomas, D.C. Addendum in Liddell, F.D.K., McDonald, J.C. & Thomas, D.C. "Methods of cohort analysis: appraisal by application to asbestos mining." *JRSS-A*, 140(4): 469-491, 1977; Langholz, B. & Goldstein, L. "Risk set sampling in epidemiologic cohort studies." *Statistical Science*, 11(1): 35-53, 1996.

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
