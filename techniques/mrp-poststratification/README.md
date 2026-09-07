# MRP -- Multilevel Regression + Poststratification (Reference §27.8)

Gelman & Little (1997); Park, Gelman & Bafumi (2004). State-of-the-
art for extrapolating from **non-probability or under-covered
samples** to a target population.

## Two-step workflow

1. **Multilevel model.** Fit the outcome as a function of covariates
   (demographics + geography) with partial pooling by cell.
2. **Poststratification.** Predict for every cell in a Census /
   registry frame; average, weighting by cell population size.

MRP corrects for **selection bias** (over-sampled cells down-weighted)
AND **small-sample noise** (rare cells shrunk toward the population).

## Files

- `python/mrp_poststratification.py` — 2-level EM for random-
  intercept multilevel model + explicit poststratification from
  scratch. Demo (G=20 cells, biased 15% sampling of cells 0-4):
  true national mean 0.379; naive sample 0.370; MRP 0.373; raw
  poststratified (no shrinkage) 0.375 — MRP retains poststratification
  accuracy while shrinking noise.
- `r/mrp_poststratification.R` — `rstanarm::stan_glmer +
  posterior_epred`, `brms`, `lme4::glmer`, `MRP`, `survey` (R);
  `pymc` / `bambi` + numpy poststratification (Python).

## When to use

- **Non-representative samples** — online panels, opt-in surveys,
  student cohorts.
- **Small-area estimation with rich cell information** — state,
  age, race, education, income.
- **Election forecasting / policy attitudes** — canonical use case
  (Gelman group).
- **Health surveys with under-covered demographics** — NHANES /
  BRFSS extension.

## When NOT to use

- **Sample is already representative** — a simple weighted mean
  suffices.
- **Cell frame not available** — you need Census / registry
  population sizes; without them, no poststratification.
- **Cells not defined by measured covariates** — MRP assumes cell
  membership is exogenous conditional on covariates.

## Assumptions & caveats

- **Cells cover the population** — every population cell has a
  known size in the poststrat frame.
- **Model correctly captures within-cell heterogeneity** — mis-
  specification biases the cell predictions.
- **Structured priors** for cells (BYM for geography, random slopes
  for interactions) improve small-cell estimation.
- **Deep MRP** uses deep learning + partial pooling — extends to
  many high-cardinality categoricals.

## Related in this repo

- `bayesian-hierarchical-models`, `generalized-linear-mixed-models`
  — the modelling side.
- `complex-survey-design`, `fay-herriot-small-area` — alternatives
  and neighbours.
- `covariate-shift-adaptation`, `transportability-generalizability`
  — the ML / causal analogues.

## Run

```
python techniques/mrp-poststratification/python/mrp_poststratification.py
Rscript techniques/mrp-poststratification/r/mrp_poststratification.R
```

**Refs:** Gelman, A. & Little, T.C. "Poststratification into many categories using hierarchical logistic regression." *Survey Methodology*, 23: 127-135, 1997; Park, D.K., Gelman, A. & Bafumi, J. "Bayesian multilevel estimation with poststratification: state-level estimates from national polls." *Political Analysis*, 12(4): 375-385, 2004.

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
