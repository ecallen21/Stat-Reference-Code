# High-Dimensional Propensity Scores (Reference §43.4)

Schneeweiss et al. (2009). Automated confounder selection from
thousands of EHR / claims codes for propensity-score estimation.
Instead of listing confounders by hand, let the data suggest them
using the Bross bias-multiplier ranking.

## Pipeline

1. **Feature generation** — binary indicators per (code, time-
   window) — diagnoses, procedures, medications, lab codes.
2. **Prevalence filter** — drop codes with prevalence < 5 % or
   > 95 %.
3. **Rank by potential confounding** — Bross multiplier:
   `|((p_1(RR−1)+1) / (p_0(RR−1)+1)) − 1|`, assuming a prior RR
   between confounder and outcome (typically 2).
4. **Fit propensity** — logistic regression of treatment on top-K
   codes + a few investigator-selected covariates.
5. **Adjust outcome analysis** — match, weight, or stratify on PS.

## When to use

- **Large claims / EHR studies** with unmeasured confounding
  concerns.
- **When investigator-selected adjustment sets** are unlikely to
  cover the true confounders.

## When NOT to use

- **Small samples** — high-dim PS estimation is unstable.
- **Randomised trials** — no need; randomisation handles
  confounding.
- **Instrumental-variable settings** — PS methods do not solve
  unmeasured-confounder problems on their own.

## Files

- `python/hdps_high_dim_propensity.py` — Bross-multiplier ranking
  + logistic PS + IPTW ATE. Demo (n=2000, p=500 sparse codes,
  5 real confounders in first 5 columns, true ATE = 2.0): naive
  diff **2.11 (biased)** → hdPS IPTW **1.93**; **5/5 true
  confounders in top 10** selected features.
- `r/hdps_high_dim_propensity.R` — `FeatureExtraction` +
  `CohortMethod` (OHDSI), `glmnet`, custom `hdps` (R);
  `sklearn.linear_model` (Python).

## Assumptions & caveats

- **No unmeasured confounding by codes** absent from the EHR —
  hdPS still requires the confounder-code relationship exists in
  data.
- **Prevalence filter cutoffs** are conventions; sensitivity
  analyses vary the thresholds.
- **Bross prior RR** affects rankings; report the value used.
- **PS-stratification / matching / IPTW** — each has its own
  assumptions; the hdPS is just the score generator.

## Related in this repo

- `propensity-score-matching` — the classical PS workflow.
- `target-trial-emulation`, `new-user-active-comparator` — full
  pharmacoepi design.
- `debiased-lasso`, `covariate-shift-adaptation` — high-dimensional
  cousins.

## Run

```
python techniques/hdps-high-dim-propensity/python/hdps_high_dim_propensity.py
Rscript techniques/hdps-high-dim-propensity/r/hdps_high_dim_propensity.R
```

**Refs:** Schneeweiss, S., Rassen, J.A., Glynn, R.J., Avorn, J., Mogun, H., & Brookhart, M.A. "High-dimensional propensity score adjustment in studies of treatment effects using health care claims data." *Epidemiology*, 2009; Rassen, J.A., Glynn, R.J., Brookhart, M.A., & Schneeweiss, S. "Covariate selection in high-dimensional propensity score analyses of treatment effects in small samples." *American Journal of Epidemiology*, 2011.

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
