# Nomograms (Reference §39.3)

Iasonos-Schrag-Raj-Panageas (2008), Harrell (2015 ch 14). Graphical
prediction tools that render a fitted regression model as a set of
**points scales** — one per predictor. Read off each predictor's
points, sum them, and map the total to a predicted probability.
Widely used in oncology staging and other bedside settings.

## Construction

For a linear predictor `η = β₀ + Σ β_j x_j`:

1. Compute the contribution range `|β_j| · (max_j − min_j)` for each
   predictor.
2. Assign the widest-contribution predictor 0-100 points.
3. Scale every other predictor's points by
   `100 · β_j / max_contrib` per unit of `x_j`.
4. Sum points across predictors → total points.
5. Convert total → linear predictor → probability via inverse link.

## When to use

- **Bedside clinical use** where the model must be evaluable
  without a computer.
- **Regulatory / journal deployment** — the standard visual form
  for logistic and Cox prediction models in oncology.
- **Interpretable communication** of coefficient magnitudes.

## When NOT to use

- **Non-additive models** (interactions, splines beyond a couple
  of knots) — the nomogram loses its clean points interpretation.
- **Very large predictor sets** — the plot becomes unreadable.
- **Uncalibrated ML models** — points from an uncalibrated boosted
  tree misrepresent probability.

## Files

- `python/nomograms.py` — build per-predictor point scales from
  logistic β and reference ranges; score two demo patients.
  Demo (cardiac-surgery mortality model): patient 1
  (age 70, creat 1.2, EF 55, no DM) → 11 pts → **P = 0.003**;
  patient 2 (age 85, creat 2.4, EF 30, DM) → 118.7 pts → **P = 0.080**.
- `r/nomograms.R` — `rms::nomogram` + `plot.nomogram`, `regplot`,
  `hdnom` (R); custom matplotlib (Python).

## Assumptions & caveats

- **Additivity** — nomograms assume `η` is additive in the
  predictors. Interactions require dedicated axes.
- **Rounding drift** — clinician-friendly integer rounding
  introduces small error; report the rounded prediction alongside
  the exact model output during validation.
- **Reference-value choice** — points are defined relative to the
  predictor's minimum; a different reference (mean, median) shifts
  the total-points scale but not predictions.
- **Deployment ≠ validation** — a nomogram is a display; the
  underlying model still needs proper bootstrap / external
  validation.

## Related in this repo

- `clinical-risk-scores` — integer point systems (Sullivan et al.
  2004) as a coarser cousin of the nomogram.
- `multivariable-model-building`, `penalized-clinical-prediction`
  — build the underlying model.
- `calibration-plots`, `discrimination-calibration` — validate the
  nomogram's predictions.

## Run

```
python techniques/nomograms/python/nomograms.py
Rscript techniques/nomograms/r/nomograms.R
```

**Refs:** Iasonos, A., Schrag, D., Raj, G.V., & Panageas, K.S. "How to build and interpret a nomogram for cancer prognosis." *Journal of Clinical Oncology*, 2008; Harrell, F.E. *Regression Modeling Strategies*, 2nd ed., Springer, 2015 (ch 14).

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
