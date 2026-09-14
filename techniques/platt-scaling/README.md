# Platt Scaling (Reference §47.282)

Platt (1999). Post-hoc calibration for BINARY classifiers by
fitting logistic regression to model scores:

```
p_calibrated = sigmoid(A · score + B)
```

Originally developed for SVMs; still standard for binary
classification with margin-based scorers. Beware of overfitting
on small calibration sets — use K-fold or a held-out split.

## Files

- `python/platt_scaling.py` — BFGS on the logistic NLL of
  (score, y). Demo: LinearSVC on 3000-sample 10-feature
  synthetic set, calibration on a 600-sample split.
  Platt-scaled Brier ≈ naive-sigmoid Brier on this well-
  behaved SVM; reliability table across 5 bins shows the
  calibrated probabilities track observed frequencies.
- `r/platt_scaling.R` — `probably::cal_estimate_logistic`,
  `e1071(probability=TRUE)` (R);
  `sklearn.calibration.CalibratedClassifierCV(method='sigmoid')`,
  `netcal.scaling.LogisticCalibration`, from-scratch
  (Python).

## When to use

- **SVM / hinge-loss classifiers** — the original setting.
- **Margin-based scorers** (gradient-boosted decision trees
  without probability output).
- **Binary classification with reasonably monotone score-vs-
  probability relationship**.

## When NOT to use

- **Non-monotone miscalibration** — Platt is a 2-parameter
  logistic; non-monotone needs isotonic / histogram binning.
- **Very small calibration sets (< 200)** — the sigmoid fit
  overfits; use Platt with cross-validation.
- **Multiclass without one-vs-rest** — Platt is binary;
  multiclass needs vector / matrix scaling.

## Assumptions & caveats

- **Monotone score-probability link** — sigmoid can only
  shift + scale; a bimodal miscalibration is invisible.
- **Held-out calibration split** — Platt on the training set
  is anti-conservative (Niculescu-Mizil & Caruana 2005 warn
  against this).
- **Cross-fitted Platt** (K-fold) is a reasonable default when
  data is scarce.
- **Reports** — always show Brier / log-loss BEFORE and AFTER,
  and a reliability curve.

## Related in this repo

- `temperature-scaling` — the multiclass analogue.
- `beta-calibration`, `histogram-binning-calibration` — more
  flexible post-hoc calibrators.
- `isotonic-regression` — non-parametric monotone alternative.
- `expected-calibration-error`, `calibration-plots` —
  diagnostics.

## Run

```
python techniques/platt-scaling/python/platt_scaling.py
Rscript techniques/platt-scaling/r/platt_scaling.R
```

**Refs:** Platt, J.C. "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods." In *Advances in Large Margin Classifiers*, MIT Press, 1999.

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
