# Discrimination vs Calibration (Reference §39.17)

Steyerberg et al. (2010), Van Calster et al. (2019). Prediction
performance has **two distinct dimensions** that a single metric can
never capture.

| Dimension | Question | Metrics |
|---|---|---|
| **Discrimination** | Can the model **rank** patients by risk? | C-statistic (AUC), c-index |
| **Calibration** | Do predicted probabilities match observed rates? | CITL, calibration slope, ICI, calibration plot |

**Discrimination is invariant to monotone rescaling of the score;
calibration is not.** Two models with identical AUC can differ
dramatically in clinical usefulness.

## Calibration metrics

- **CITL** = `mean(y) − mean(p̂)`; ideal 0.
- **Slope** = slope of `y ~ logit(p̂)`; ideal 1.
- **ICI** (Austin-Steyerberg 2019) — mean absolute distance between
  predicted probability and locally-smoothed observed rate.
- **Brier score** = `mean((p̂ − y)²)`; proper score combining both
  dimensions.

Van Calster: "Calibration is the Achilles heel of predictive
analytics."

## When to use

- **Every clinical prediction model** report should carry both
  discrimination and calibration metrics.
- **ML model deployment** — trees and neural nets often
  discriminate well but calibrate poorly; recalibrate before use.

## When NOT to use

- **Pure ranking tasks** where the absolute probability is
  irrelevant (some triage / referral prioritisation contexts).

## Files

- `python/discrimination_calibration.py` — AUC, CITL, slope,
  Brier, ICI. Demo: two models with **identical AUC = 0.705** on
  the same data — Model A (correct logit) CITL 0.006, slope 0.835,
  Brier 0.219, ICI 0.040; Model B (scaled + shifted logit) CITL
  0.136, slope 1.193, Brier 0.239, ICI 0.136. Same discrimination,
  very different calibration.
- `r/discrimination_calibration.R` — `rms::validate`/`val.prob`,
  `pROC`, `CalibrationCurves` (R); `sklearn.metrics` + custom
  (Python).

## Assumptions & caveats

- **Report both** at model launch and at every revalidation.
- **Metric family** — always report a proper score (Brier / log-
  loss) as the summary, plus discrimination and calibration
  separately for interpretation.
- **AUC is not enough** — a model can be perfectly discriminating
  but wildly miscalibrated, useless for absolute-risk decisions.
- **Calibration plot** (`calibration-plots`) visualises what the
  slope + CITL summarise.

## Related in this repo

- `calibration-plots` — visual companion metric.
- `external-validation`, `bootstrap-optimism-correction` — apply
  these metrics honestly.
- `model-recalibration` — the fix when calibration drifts.

## Run

```
python techniques/discrimination-calibration/python/discrimination_calibration.py
Rscript techniques/discrimination-calibration/r/discrimination_calibration.R
```

**Refs:** Steyerberg, E.W., Vickers, A.J., Cook, N.R., et al. "Assessing the performance of prediction models: a framework for traditional and novel measures." *Epidemiology*, 2010; Van Calster, B., McLernon, D.J., van Smeden, M. et al. "Calibration: the Achilles heel of predictive analytics." *BMC Medicine*, 2019; Austin, P.C. & Steyerberg, E.W. "The integrated calibration index (ICI)." *Statistics in Medicine*, 2019.

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
