# Data-Drift Detection (Reference Ch 32 MLOps)

Detect changes in the **input distribution `p(x)`** between a
**reference** window (usually training data) and a **current** window
(production traffic). Three canonical univariate scores + a multivariate
extension.

## PSI — Population Stability Index (Wu-Olson 2010)

```
PSI = Σ_b ( q_b − p_b ) · log( q_b / p_b )
```

with `p_b`, `q_b` the bin proportions of reference / current. Standard
thresholds:

- `PSI < 0.10` — no significant drift.
- `0.10 ≤ PSI < 0.25` — moderate drift.
- `PSI ≥ 0.25` — severe drift; investigate.

## KS — Two-sample Kolmogorov-Smirnov

```
D  =  sup_x  | F_ref(x) − F_cur(x) |
```

Distribution-free; independent of feature scale.

## Wasserstein-1 (Earth-mover)

```
W₁ = ∫ | F_ref⁻¹(u) − F_cur⁻¹(u) | du
```

In the feature's original units — interpretable as "how far, on
average, does the distribution have to move?".

## Multivariate extensions (Rabanser 2019)

- **MMD** (Maximum Mean Discrepancy) with an RBF kernel.
- **Learned domain-classifier** — logistic regression separating
  reference vs current; AUROC substantially > 0.5 ⇒ drift.
- **Aggregation** of per-feature p-values with Bonferroni or FDR
  (`Benjamini-Hochberg`).

## When to use

- **Any deployed model** — cheap to compute; run daily / hourly.
- **Explains** performance degradation (label drift needs
  `concept-drift-adwin` on top).
- **Compliance** — regulators increasingly ask for population-stability
  reports on any consumer-scoring model.

## When NOT to use

- **High-cardinality categorical** features — bin sensibly first.
- **Deliberate feature evolution** (holiday season, product launch) —
  alerts will fire even without model degradation.
- **Small windows** — the scores are noisy for `n < 500`.

## Files

- `python/data_drift_detection.py` — from-scratch `psi`, `ks_statistic`,
  `wasserstein1`, plus a `per_feature_report`. Demo on synthetic
  reference vs shifted current data: **x0 unchanged → PSI 0.008 (OK);
  x1 shifted mean +0.4 → PSI 0.200 (moderate); x2 variance ×2 → PSI
  0.531 (SEVERE)**.
- `r/data_drift_detection.R` — `driftR` / `drifter` (R); `evidently` /
  `alibi-detect` / `whylogs` / `nannyML` (Python).

## Assumptions & caveats

- **PSI thresholds are conventional**, not universal — calibrate on
  historical stability of your own features.
- **Binning strategy matters** — quantile-bins are more robust than
  equal-width; use `n_bins ∈ [5, 20]`.
- **Zero-count bins** need smoothing (`+ ε`) or PSI blows up.
- **Univariate only** — a joint distribution can drift even when
  every marginal is stable; use MMD or a domain-classifier for the
  joint audit.
- **Not a substitute for label-based monitoring** — see
  `concept-drift-adwin` and `model-monitoring-metrics`.

## Related in this repo

- `concept-drift-adwin` — label / performance drift.
- `model-monitoring-metrics` — rolling accuracy + calibration.
- `covariate-shift-adaptation` — mitigation via density-ratio weighting.
- `kolmogorov-smirnov`, `wasserstein-distance` (if present) — the
  general statistical tests.
- `ood-detection` — per-example OOD is a per-input version of the
  same idea.

## Run

```
python techniques/data-drift-detection/python/data_drift_detection.py
Rscript techniques/data-drift-detection/r/data_drift_detection.R
```

**Refs:** Wu, D. & Olson, D. "A comparison of stability measures for financial time series." *Journal of Risk Model Validation*, 2010; Rabanser, S., Günnemann, S. & Lipton, Z. "Failing loudly: an empirical study of methods for detecting dataset shift." *NeurIPS*, 2019.

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
