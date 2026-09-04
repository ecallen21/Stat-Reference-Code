# Clinical Risk Scores (Reference §39.12)

Sullivan-Massaro-D'Agostino (2004), Moons et al. (2009). Convert a
fitted regression model into an **integer point** scoring system so
clinicians can compute risk at the bedside. This template underlies
Framingham, CHA₂DS₂-VASc, Wells, APACHE II, SOFA, QRISK, and dozens
of other named scores.

## Sullivan et al. method

1. **Choose a reference value** `W_ref_j` for each predictor
   (often the healthiest category).
2. **Compute contribution** `β_j · (W_j − W_ref_j)` for every level.
3. **Choose units-per-point** `B` so the widest predictor spans a
   memorable range (e.g. 0-6 points).
4. **Round to integers** — `points_j = round( contribution_j / B )`.
5. **Sum** across categories → total score → predicted probability
   via `logit⁻¹(β₀ + total · B)`.

## When to use

- **Bedside deployment** without a calculator.
- **Journal presentation** — the standard visual form for risk
  models in cardiology, ID, oncology, obstetrics.
- **Regulatory / EHR embedding** where a fixed integer table beats
  a floating-point calculation.

## When NOT to use

- **High-dimensional / continuous-predictor** models — rounding
  drops too much information.
- **Interactions / nonlinearities** — a simple additive score
  cannot represent them accurately.
- **When calibration precision matters** — always report the
  nomogram / continuous model alongside the rounded score.

## Files

- `python/clinical_risk_scores.py` — Sullivan integer-points
  scoring from logistic β and reference levels. Demo (5-predictor
  CHA₂DS₂-VASc-style toy): B = 0.21; scoring table `age_group
  {<65:+0, 65-74:+3, ≥75:+6}, prior_stroke {no:+0, yes:+5}, …`.
  Patient scoring: 8 pts → **P(stroke) = 0.14**, 20 pts → **0.67**.
- `r/clinical_risk_scores.R` — `rms::nomogram` +
  `rms::points.chart`, `AutoScore` (R); custom (Python).

## Assumptions & caveats

- **Rounding drift** — always publish the max absolute deviation
  between rounded and continuous predictions.
- **Calibration on target population** — a rounded score derived on
  cohort A must be recalibrated in cohort B (same as any model).
- **Missing categories** — decide up-front whether "missing" gets
  its own points row or is imputed.
- **Reference-value choice** shifts total-points scale but not
  predictions; document it explicitly.

## Related in this repo

- `nomograms` — the continuous visual cousin of the integer
  score.
- `multivariable-model-building` — build the underlying model.
- `external-validation`, `model-recalibration` — check and fix
  performance in new cohorts.

## Run

```
python techniques/clinical-risk-scores/python/clinical_risk_scores.py
Rscript techniques/clinical-risk-scores/r/clinical_risk_scores.R
```

**Refs:** Sullivan, L.M., Massaro, J.M., & D'Agostino, R.B. "Presentation of multivariate data for clinical use: the Framingham Study risk score functions." *Statistics in Medicine*, 2004; Moons, K.G.M., Royston, P., Vergouwe, Y., Grobbee, D.E., & Altman, D.G. "Prognosis and prognostic research: what, why, and how?" *BMJ*, 2009; Xie, F. et al. "AutoScore." *JMIR Medical Informatics*, 2020.

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
