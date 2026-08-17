# Probability Calibration + Scaling (Reference §26.15)

A classifier that predicts `p(y = 1 | x) = 0.9` is **well-calibrated** iff among all inputs where it predicts 0.9, roughly 90% truly have `y = 1`.

## Diagnostics

- **Reliability diagram** — bin predictions by predicted probability; plot mean predicted vs empirical positive rate. Diagonal = perfect calibration.
- **Brier score** — mean `(p − y)²`.
- **Expected calibration error (ECE)** — weighted mean bin-wise miscalibration.
- **Log loss** — proper scoring rule.

## Post-hoc calibration methods (fit on a held-out calibration set)

- **Platt scaling** (Platt 1999): `p_cal = sigmoid(a · s + b)` — good for SVM / boosting.
- **Isotonic regression** — nonparametric monotone map (see `isotonic-regression`); needs more calibration data.
- **Temperature scaling** — single scalar `T` for neural nets: `softmax(z / T)`.
- **Beta calibration** — more flexible than Platt on 3-parameter Beta distribution shape.

## Files

- `python/calibration_scaling.py` — Brier + log-loss + reliability diagram + Platt + isotonic calibrators (isotonic reuses the `isotonic-regression` module). Demo: raw over-confident scores ECE 0.019 → Platt 0.015 / Isotonic 0.017.
- `r/calibration_scaling.R` — base R Platt via `glm(family = binomial)` + isotonic via `isoreg` + `rms::val.prob` for the standard calibration diagnostics.

## When to calibrate

- **Predictions used in decisions with cost/utility** — expected loss depends on well-calibrated probabilities.
- **Combining models** with different scale ranges.
- **Any tree / SVM / kNN classifier** — these tend to be miscalibrated by default.

## When it's less important

- Only the **argmax class** matters (accuracy metric), not the probability itself.
- Naive Bayes probability estimates are usually meaningless anyway; use the ranking.

## Choose Platt vs isotonic

- **Small calibration set** (`n_cal < 1000`) → Platt (2 parameters, more stable).
- **Larger calibration set** → isotonic (nonparametric, adapts to any monotone miscalibration).

## Run

```
python techniques/calibration-scaling/python/calibration_scaling.py
Rscript techniques/calibration-scaling/r/calibration_scaling.R
```

**Refs:** Platt, J.C. "Probabilistic outputs for support vector machines." *Adv. Large Margin Classifiers* 10(3), 61–74, 1999; Zadrozny, B. & Elkan, C. "Transforming classifier scores into accurate multiclass probability estimates." *KDD*, 2002; Guo, C., Pleiss, G., Sun, Y. & Weinberger, K.Q. "On calibration of modern neural networks." *ICML*, 2017.

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
