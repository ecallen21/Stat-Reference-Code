# Calibration Plots (Reference §39.19)

Van Calster et al. (2019), Austin & Steyerberg (2019). Plot
**observed** event rate against **predicted** probability. Perfect
calibration is the 45° diagonal.

## Two constructions

- **Grouped (decile) plot** — bin predictions into 10 groups, plot
  observed rate vs mean predicted per group. Simple; sensitive to
  bin count and boundary artefacts.
- **Smoothed** (LOESS / spline) — nonparametric local regression of
  `y` on `p̂`. Steyerberg (2019) recommends over binning.

## Numerical summaries

- **ICI** (Integrated Calibration Index) — mean absolute distance
  between predicted probability and the smoother; single-number
  summary.
- **E-max** — worst-case absolute miscalibration across the
  probability range.
- **E-90** — 90th-percentile absolute miscalibration.

## When to use

- **Every prediction model** at development, internal validation,
  and external validation.
- **Post-recalibration** to verify the fix worked.

## When NOT to use

- **Extremely rare-outcome models** where decile bins have zero
  events — use pooled bins or Bayesian smoothing.

## Files

- `python/calibration_plots.py` — decile-grouped and LOESS
  calibration curves + ICI / E-max / E-90. Demo (n=800): well-
  calibrated model **ICI = 0.015, E-max = 0.027**; over-confident
  model (scaled logit) **ICI = 0.082, E-max = 0.143**.
- `r/calibration_plots.R` — `rms::calibrate`,
  `rms::val.prob`, `CalibrationCurves::val.prob.ci.2`,
  `predtools::calibration_plot`, `pmcalibration` (R);
  `sklearn.calibration.calibration_curve` + custom (Python).

## Assumptions & caveats

- **Group count** — 10 deciles is convention; sensitive to
  boundaries. LOESS is safer.
- **Bootstrap the curve** for confidence bands (`rms::calibrate`
  does this by default).
- **Report ICI + E-max**, not just a picture. A pretty plot with
  no numerical summary is useless in a paper.
- **Recalibrate then re-plot** — the diagnostic must be repeated
  after any recalibration step.

## Related in this repo

- `discrimination-calibration` — parametric summaries.
- `external-validation`, `model-recalibration` — the workflow this
  plot fits into.

## Run

```
python techniques/calibration-plots/python/calibration_plots.py
Rscript techniques/calibration-plots/r/calibration_plots.R
```

**Refs:** Van Calster, B., McLernon, D.J., van Smeden, M. et al. "Calibration: the Achilles heel of predictive analytics." *BMC Medicine*, 2019; Austin, P.C. & Steyerberg, E.W. "The integrated calibration index (ICI) and related metrics for quantifying the calibration of logistic regression models." *Statistics in Medicine*, 2019; Steyerberg, E.W. *Clinical Prediction Models*, 2nd ed., Springer, 2019 (ch 15).

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
