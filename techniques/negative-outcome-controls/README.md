# Negative Outcome Controls + Empirical Calibration (Reference §43.9)

Schuemie et al. (2014), Lipsitch et al. (2010). Choose outcomes
**known not to be affected** by the exposure ("negative controls").
Under no residual confounding they should give effect estimates
centred on zero (`log HR = 0`).

Deviations from zero quantify **systematic error**. Fit an
**empirical null** distribution to those negative-control effect
sizes, then recalibrate the p-values of the true outcome analysis.

## Empirical null (Schuemie 2014)

```
log HR_i ~ N(μ_null, τ² + SE_i²)
```

Estimate `(μ_null, τ)` by MLE from `k` negative-control estimates
and their SEs. Recompute the calibrated p-value for the outcome of
interest as

```
z_cal = (log HR − μ_null) / √(SE² + τ²)
p_cal = 2 · Φ̄(|z_cal|)
```

## When to use

- **Observational studies** with unmeasured confounding concerns.
- **OHDSI-style large-scale pharmacoepi** where dozens of negative
  controls can be selected automatically.
- **Sensitivity analyses** — report both naive and calibrated
  effect measures.

## When NOT to use

- **Randomised trials** — no residual confounding to calibrate.
- **Too few negative controls** — Schuemie recommends ≥ 30.
- **Negative controls affected by exposure** — invalid; the whole
  method depends on their true null.

## Files

- `python/negative_outcome_controls.py` — MLE of the empirical
  null + calibrated p (custom). Demo (40 negative controls with a
  systematic +0.15 bias): estimated `μ_null = 0.22, τ = 0.22`.
  True outcome `log HR = 0.4, SE = 0.12`: **naive p = 8.6 × 10⁻⁴**
  → **calibrated p = 0.47** — the naive "significant" result is a
  systematic-error artefact.
- `r/negative_outcome_controls.R` — `EmpiricalCalibration::fitNull`
  + `calibrateP`, `Cyclops`, `MethodEvaluation` (OHDSI) (R);
  custom (Python).

## Assumptions & caveats

- **Negative-control choice** must be transparent and pre-specified;
  post-hoc cherry-picking invalidates the calibration.
- **Same design** — negative-control estimates must come from the
  same estimation pipeline (design, model, covariates) as the
  outcome of interest.
- **Normal approximation** — the null is modelled as normal; heavy-
  tailed systematic error breaks it.
- **Complementary tools** — negative controls detect *residual*
  confounding, not selection bias or measurement error.

## Related in this repo

- `hdps-high-dim-propensity`, `target-trial-emulation` — companion
  bias-reduction tools.
- `multiple-testing-corrections`, `false-discovery-rate` — a
  different kind of p-value adjustment.

## Run

```
python techniques/negative-outcome-controls/python/negative_outcome_controls.py
Rscript techniques/negative-outcome-controls/r/negative_outcome_controls.R
```

**Refs:** Schuemie, M.J., Ryan, P.B., DuMouchel, W., Suchard, M.A., & Madigan, D. "Interpreting observational studies: why empirical calibration is needed to correct p-values." *Statistics in Medicine*, 2014; Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. "Negative controls: a tool for detecting confounding and bias in observational studies." *Epidemiology*, 2010.

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
