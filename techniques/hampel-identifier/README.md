# Hampel Identifier (Reference §47.375)

Hampel (1971); Hampel-Ronchetti-Rousseeuw-Stahel (1986). Flag
point `x_i` as an outlier when

```
|x_i − median| > k · 1.4826 · MAD
```

with `MAD = median(|x − median|)` and factor `1.4826` making
the scale Fisher-consistent at Gaussian. Threshold `k = 3` is
the robust 3-sigma rule. Extends to sliding-window form: at
each index compute the local median / MAD and replace an
outlier by the local median (Hampel filter).

## Files

- `python/hampel_identifier.py` — Two demos. (1) Static
  n=200 Gaussian sample with four wild outliers at
  `[10, 40, 100, 150]`: all four detected (plus one
  genuine-tail 188). (2) Sliding-window filter on sinusoidal
  signal with 15 injected spikes: RMSE drops from 0.63
  before to 0.10 after replacement (6× cleaner).
- `r/hampel_identifier.R` — `pracma::hampel`,
  `seewave::hampel_filter` (R); `scipy.signal.medfilt` (median
  filter cousin), from-scratch (Python).

## When to use

- **Robust 1-D outlier detection** — cleaner alternative to
  `mean ± 3 σ`.
- **Time-series pre-processing** — spike removal in EEG,
  vibration, financial-tick streams.
- **Quality control** — MAD-based robust limits for control
  charts.

## When NOT to use

- **Multivariate data** — use MCD / Mahalanobis distance.
- **When the true distribution is highly skewed** — median-
  MAD is still valid but efficiency drops; use medcouple.
- **Very short windows** — MAD estimate has high variance;
  use larger window or global Hampel.

## Assumptions & caveats

- **Fisher-consistency at Gaussian** — factor `1.4826 =
  1 / Φ⁻¹(0.75)` for Gaussian; different scales apply for
  other assumed distributions.
- **k threshold** — `k = 3` is common; `k = 4.5` for looser
  detection.
- **Boundary effects** — sliding window shrinks at edges;
  use partial windows or reflect padding.
- **Replacement vs flag-only** — deleting or replacing
  contaminates downstream time-series continuity; some
  workflows only flag.
- **Zero-MAD edge cases** — MAD = 0 when > half of the
  window is identical; return no outliers or fall back to
  interquartile scale.

## Related in this repo

- `least-trimmed-squares`, `mcd-robust-covariance`,
  `mm-estimators-robust`, `winsorization`,
  `grubbs-outlier-test`, `outlier-tests` — robust /
  outlier neighbours.
- `robust-location-scale`, `savitzky-golay-filter`,
  `butterworth-bandpass` — signal cleaning cousins.
- `isolation-forest-anomaly`, `one-class-svm`,
  `ts-anomaly-detection` — modern anomaly-detection
  alternatives.

## Run

```
python techniques/hampel-identifier/python/hampel_identifier.py
Rscript techniques/hampel-identifier/r/hampel_identifier.R
```

**Refs:** Hampel, F.R. "A general qualitative definition of robustness." *Ann. Math. Statist.*, 42: 1887-1896, 1971; Hampel, F.R., Ronchetti, E.M., Rousseeuw, P.J. and Stahel, W.A. *Robust Statistics: The Approach Based on Influence Functions*, Wiley, 1986.

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
