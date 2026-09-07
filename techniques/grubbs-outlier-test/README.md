# Grubbs' + Rosner ESD Outlier Tests (Reference §3.26)

Grubbs (1950); Rosner (1983). Formal tests that the most extreme
observation(s) are outliers under a Gaussian null.

## Grubbs (single outlier)

    G = max_i |x_i − x̄| / s

Reject at level α if

    G > ((n − 1) / √n) · √(t² / (n − 2 + t²)),  t = t_{α/(2n), n−2}

## Rosner ESD (up to k outliers)

Iteratively delete the most-extreme observation and compute `R_i`;
compare to `λ_i` critical values (Rosner 1983). The number of
outliers is the largest `i` such that `R_i > λ_i`.

## Files

- `python/grubbs_outlier_test.py` — Grubbs' two-sided + Rosner ESD
  from scratch. Demo: 30 clean N(10, 1) points + injected outlier
  at 20 → Grubbs G = 4.92 > G_crit = 2.92 (flagged); 3 injected
  outliers (20, 21, 22) → Rosner correctly detects 3.
- `r/grubbs_outlier_test.R` — `outliers::grubbs.test /
  dixon.test`, `EnvStats::rosnerTest`, `car::outlierTest` (R);
  from-scratch (Python).

## When to use

- **Small-to-moderate n, Gaussian data** — Grubbs is standard in
  analytical chemistry, engineering QC.
- **Known number of suspects** (≤ ~10) — Rosner ESD.
- **Prescreening before regression / assay analysis**.

## When NOT to use

- **Non-Gaussian residuals** — Grubbs over-rejects heavy-tailed data.
- **Multivariate outliers** — use Mahalanobis / Isolation Forest.
- **Large n** — everything looks extreme; consider practical
  thresholds (IQR × 1.5, robust-Z > 3.5).
- **Time-series with outliers-in-innovations** — use robust ARIMA /
  X-11 diagnostics.

## Assumptions & caveats

- **Normality** — validate with QQ-plot before applying.
- **Independence** — Grubbs assumes iid; violated for time-series.
- **Masking / swamping** — one outlier can hide another; ESD /
  Rosner remedy this within limits.
- **Multiple testing** — iterating Grubbs and re-running raises
  α; use Bonferroni or Rosner directly.

## Related in this repo

- `outlier-tests`, `multivariate-outlier-detection` — outlier
  family.
- `robust-regression`, `robust-location-scale`, `mm-estimators-robust`
  — robust alternatives.
- `isolation-forest-anomaly`, `one-class-svm` — ML anomaly detection.
- `winsorization` — cousin robust processing.

## Run

```
python techniques/grubbs-outlier-test/python/grubbs_outlier_test.py
Rscript techniques/grubbs-outlier-test/r/grubbs_outlier_test.R
```

**Refs:** Grubbs, F.E. "Sample criteria for testing outlying observations." *Annals of Mathematical Statistics*, 21(1): 27-58, 1950; Rosner, B. "Percentage points for a generalized ESD many-outlier procedure." *Technometrics*, 25(2): 165-172, 1983.

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
