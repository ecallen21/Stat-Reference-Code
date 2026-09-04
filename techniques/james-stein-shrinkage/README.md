# James-Stein Shrinkage Estimation (Reference §38.15)

Stein (1956), James & Stein (1961). When estimating `p ≥ 3` normal
means simultaneously, the sample-mean vector is **inadmissible**: a
convex combination that shrinks toward a common target has strictly
smaller total mean-squared error, for every true `θ`.

## Estimator (positive-part JS toward grand mean)

For `y_i ~ N(θ_i, σ²)`, `i = 1, ..., p`,

```
θ̂_JS_i = ȳ + max( 0, 1 − (p − 3) σ² / Σ (y_j − ȳ)² ) · (y_i − ȳ)
```

- `(p − 3)` becomes positive for `p ≥ 4`; for `p = 3` the classical
  JS coefficient degenerates and no shrinkage occurs.
- The **positive-part** rule prevents sign-flipping over-shrinkage.

## Why it matters

This launched **empirical Bayes**, the entire modern shrinkage
programme (LASSO, ridge, hierarchical models), and small-area
estimation.

## When to use

- **Small-area / small-group means** — school-, hospital-,
  neighbourhood-level averages.
- **Multi-arm trials** with many treatment comparisons.
- **Ranked-list estimation** — batting averages, teacher effects,
  hospital league tables.

## When NOT to use

- **`p ≤ 3`** — no dominance available.
- **Single well-motivated target** — a hierarchical Bayes model is
  more transparent.
- **Very unequal `σ_i²`** — use the weighted / Efron-Morris JS with
  a common target.

## Files

- `python/james_stein_shrinkage.py` — positive-part JS toward the
  grand mean, compared to MLE via total MSE over 400 simulations at
  `p ∈ {3, 5, 10, 25, 50}`. Demo: **JS / MLE ratio drops to 0.47
  at p=25**; larger `p` gives larger gain.
- `r/james_stein_shrinkage.R` — no dedicated JS package; `corpcor`
  for multivariate covariance shrinkage; custom (Python).

## Assumptions & caveats

- **`σ²` known or well-estimated** — a bad `σ̂²` sinks the gain.
- **Common target** — shrinking toward the grand mean is
  conventional but arbitrary; any fixed target dominates the MLE.
- **Per-component MSE trade-off** — JS reduces total MSE but any
  single `θ_i` estimate can be worse than its MLE.
- **Bias** — JS is biased; bias-variance trade-off is favourable in
  aggregate but not for a targeted individual estimate.

## Related in this repo

- `covariance-estimation-highdim` — Ledoit-Wolf shrinkage covariance
  is the multivariate cousin.
- `bayesian-glms`, `hierarchical-models` — modern shrinkage
  homes.
- `empirical-bayes` (if present) — direct descendant.

## Run

```
python techniques/james-stein-shrinkage/python/james_stein_shrinkage.py
Rscript techniques/james-stein-shrinkage/r/james_stein_shrinkage.R
```

**Refs:** Stein, C. "Inadmissibility of the usual estimator for the mean of a multivariate normal distribution." *Proc. Berkeley Symp. Math. Statist. Prob.*, 1956; James, W. & Stein, C. "Estimation with quadratic loss." *Proc. Berkeley Symp.*, 1961; Efron, B. & Morris, C. "Stein's estimation rule and its competitors." *JASA*, 1973; Efron, B. *Large-Scale Inference*, Cambridge University Press, 2010.

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
