# Cox Regression with Time-Varying Covariates (Reference §11.10)

Therneau & Grambsch (2000). Cox model where covariate values
change over follow-up:

```
λ(t | x) = λ₀(t) · exp(β · x(t))
```

Handled via **counting-process** rows `(start, stop, event, x)`.
Each subject with time-varying x contributes multiple rows split
at each value change. Partial-likelihood at each event time uses
the covariate value at that event time.

## When to use

- **Exposure switches during follow-up** — treatment initiations,
  test results, biomarker updates.
- **Landmark analysis avoidance** — retain all information rather
  than discarding pre-landmark events.
- **Correct handling of immortal-time bias** (see companion
  technique).

## When NOT to use

- **Outcome affects the covariate** — reverse causation biases
  the estimate.
- **Reciprocal feedback** between covariate and hazard — MSM /
  g-methods are needed.

## Files

- `python/cox_time_varying.py` — long-format partial likelihood
  with a single time-varying covariate (custom). Demo (n=500,
  covariate switches 0→1 at random time in [1, 5], true β=0.7):
  **estimated β̂ = 0.528, HR = 1.70**.
- `r/cox_time_varying.R` — `survival::coxph` +
  `Surv(start, stop, event)` + `tmerge`, `timereg` (R);
  `lifelines::CoxTimeVaryingFitter`, `scikit-survival` (Python).

## Assumptions & caveats

- **PH assumption** at each time — check Schoenfeld residuals.
- **Row-splitting must be exhaustive** — every covariate change
  needs a new row.
- **External vs internal covariates** — external (calendar time,
  environmental) are unproblematic; internal (biomarkers affected
  by the disease) need g-methods.
- **Confidence intervals** via the observed information; robust
  sandwich for clustered data.

## Related in this repo

- `immortal-time-bias`, `landmark-analysis`,
  `marginal-structural-model` — related time-dependent analysis.
- `parametric-survival`, `additive-aalen` — other survival
  models.

## Run

```
python techniques/cox-time-varying/python/cox_time_varying.py
Rscript techniques/cox-time-varying/r/cox_time_varying.R
```

**Refs:** Therneau, T.M. & Grambsch, P.M. *Modeling Survival Data: Extending the Cox Model*, Springer, 2000; Fisher, L.D. & Lin, D.Y. "Time-dependent covariates in the Cox proportional hazards regression model." *Annual Review of Public Health*, 1999.

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
