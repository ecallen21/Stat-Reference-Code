# Expected Calibration Error (Reference §47.283)

Naeini, Cooper & Hauskrecht (2015); Guo et al (2017). Weighted
average absolute gap between confidence and accuracy across
confidence bins:

```
ECE = Σ_m (|B_m| / n) · |acc(B_m) − conf(B_m)|
MCE = max_m |acc(B_m) − conf(B_m)|
ACE = adaptive (equal-mass) binning ECE
```

Standard metric for calibration. Complements NLL / Brier and
reliability diagrams.

## Files

- `python/expected_calibration_error.py` — Fixed-width and
  adaptive-binning ECE + MCE. Demo: overconfident model
  (n=4000, confs = 1.5 × true prob) gives ECE ≈ 0.16 and
  MCE ≈ 0.32; well-calibrated (confs = true prob) gives
  ECE ≈ 0.02 (residual noise). Adaptive binning matches
  the equal-width result on this uniform distribution.
- `r/expected_calibration_error.R` — `probably::cal_plot_windowed`,
  `yardstick::brier_class`, `Metrics::logLoss` (R);
  `netcal.metrics.ECE`, `torchmetrics.CalibrationError`,
  from-scratch (Python).

## When to use

- **Reporting calibration** in ML papers — ECE is the
  standard.
- **Comparing calibration methods** (temperature, Platt,
  Beta, histogram binning).
- **Model selection** — pair ECE with accuracy for
  reliability-quality trade-off.

## When NOT to use

- **Sole calibration metric** — ECE has known pitfalls (bin
  choice, weighting); pair with reliability plots and NLL /
  Brier.
- **Very small n** — with few samples per bin, ECE is noisy;
  use fewer bins or adaptive binning.
- **Debiased inference on ECE** — the sample-average ECE is
  a biased estimator; Roelofs et al 2022 debiased ECE (BE-ECE)
  or MMCE.

## Assumptions & caveats

- **Bin count `M`** — 10-15 is standard for equal-width; 10
  is fine for equal-mass.
- **Equal-width vs adaptive** — equal-width is easier to read
  in a table; adaptive avoids near-empty bins at the tails.
- **Multiclass ECE** — apply on max-probability by default;
  class-wise ECE is more informative.
- **MCE vs ECE** — MCE emphasises the worst bin (safety),
  ECE the average (reporting).

## Related in this repo

- `temperature-scaling`, `platt-scaling`, `beta-calibration`,
  `histogram-binning-calibration` — post-hoc calibrators.
- `calibration-plots`, `discrimination-calibration` — visual
  companions.
- `cross-entropy-log-loss`, `proper-scoring-rules-crps` —
  proper scoring alternatives.

## Run

```
python techniques/expected-calibration-error/python/expected_calibration_error.py
Rscript techniques/expected-calibration-error/r/expected_calibration_error.R
```

**Refs:** Naeini, M.P., Cooper, G.F. and Hauskrecht, M. "Obtaining well calibrated probabilities using Bayesian binning." In *AAAI*, 2015; Guo, C. et al. "On calibration of modern neural networks." In *ICML*, 2017.

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
