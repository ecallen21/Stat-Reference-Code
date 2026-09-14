# Histogram Binning Calibration (Reference §47.285)

Zadrozny & Elkan (2001). Non-parametric calibration for
BINARY classifiers: partition scores into `M` bins, estimate

```
p_calibrated(s ∈ bin m) = fraction of positives in that bin
```

Simple and effective when sample size per bin is large. Bin
choice (equal-width vs equal-mass vs Bayesian-optimal) is the
key knob.

## Files

- `python/histogram_binning_calibration.py` — Adaptive
  (equal-mass) and equal-width binning. Demo: non-monotone
  miscalibration `s = p + 0.15 sin(6π p)` on n=5000. Brier
  raw 0.24, drops to ~0.19 with M=10 adaptive bins. Reliability
  table shows observed rates tracking scaled bin midpoints.
- `r/histogram_binning_calibration.R` —
  `probably::cal_estimate_isotonic`, custom `cut + tapply` (R);
  `netcal.binning.HistogramBinning`, `sklearn` isotonic,
  from-scratch (Python).

## When to use

- **Non-monotone miscalibration** — where Platt / Beta fail.
- **Large calibration set** — bins need enough samples per
  bin for stable rates.
- **Interpretable calibration** — the bin-wise rates are a
  transparent lookup table.

## When NOT to use

- **Small calibration set** — few samples per bin means noisy
  rates; use Beta / Platt.
- **Where continuity matters** — histogram binning has step
  discontinuities at bin boundaries; use isotonic for a
  monotone step-free calibrator.
- **Multiclass without one-vs-rest** — apply per class.

## Assumptions & caveats

- **Bin count `M`** — 10-20 adaptive bins is standard.
- **Equal-mass vs equal-width** — equal-mass avoids empty
  bins; equal-width is easier to interpret.
- **Bayesian binning** (BBQ, Naeini 2015) averages over
  binning schemes for smoother calibrated probabilities.
- **Held-out fit** — cross-fitted binning helps when n is
  moderate.

## Related in this repo

- `platt-scaling`, `beta-calibration`, `temperature-scaling`
  — parametric alternatives.
- `isotonic-regression` — monotone non-parametric alternative.
- `expected-calibration-error`, `calibration-plots` —
  companion evaluation.

## Run

```
python techniques/histogram-binning-calibration/python/histogram_binning_calibration.py
Rscript techniques/histogram-binning-calibration/r/histogram_binning_calibration.R
```

**Refs:** Zadrozny, B. and Elkan, C. "Obtaining calibrated probability estimates from decision trees and naive Bayesian classifiers." In *ICML*, pp. 609-616, 2001; Zadrozny, B. and Elkan, C. "Transforming classifier scores into accurate multiclass probability estimates." In *KDD*, 2002.

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
