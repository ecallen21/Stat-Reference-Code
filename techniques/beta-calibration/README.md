# Beta Calibration (Reference §47.284)

Kull, Silva Filho & Flach (2017). More expressive alternative
to Platt scaling for binary classifiers:

```
p_calibrated = sigmoid(a · log(s) − b · log(1 − s) + c)
```

Equivalent to logistic on (log s, −log(1 − s), 1). Recovers
Platt as a special case (a = b) and captures ASYMMETRIC
miscalibration.

## Files

- `python/beta_calibration.py` — BFGS on the beta-family NLL
  plus Platt for comparison. Demo: asymmetrically overconfident
  model (n=3000 with inflation only on negatives). Raw Brier
  0.36 → Platt 0.25 → Beta 0.22 (Beta wins by 12% on
  asymmetric miscalibration).
- `r/beta_calibration.R` — `betacal` (R port),
  `probably::cal_estimate_beta` (R); `betacal` (PyPI),
  `netcal.scaling.BetaCalibration`, from-scratch (Python).

## When to use

- **Asymmetric miscalibration** — one tail systematically
  over- or under-confident.
- **Small validation sets** — Beta has 3 params, still stable
  where isotonic overfits.
- **Modern default** in many AutoML pipelines.

## When NOT to use

- **Non-monotone miscalibration** — histogram / isotonic
  captures arbitrary shapes.
- **Predictions all near 0 or 1** — log-log link becomes
  numerically unstable; clip scores.
- **Multiclass** — Beta is binary; multiclass needs per-class
  application.

## Assumptions & caveats

- **Score range** — scores must be in (0, 1); clip or squash
  logits with sigmoid first.
- **Held-out fitting** — beta-calibration on the training set
  is anti-conservative.
- **Sub-case a = b** — reduces to Platt; check whether the
  extra degree of freedom is warranted (AIC / likelihood-
  ratio).
- **Cross-fitted Beta** helps with small n.

## Related in this repo

- `platt-scaling` — the 2-parameter cousin.
- `temperature-scaling` — the multiclass analogue.
- `histogram-binning-calibration`, `isotonic-regression` —
  non-parametric alternatives.
- `expected-calibration-error` — evaluation metric.

## Run

```
python techniques/beta-calibration/python/beta_calibration.py
Rscript techniques/beta-calibration/r/beta_calibration.R
```

**Refs:** Kull, M., Silva Filho, T.M. and Flach, P. "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers." In *AISTATS*, pp. 623-631, 2017.

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
