# Higher Criticism (Reference §47.376)

Donoho & Jin (2004). Global-null test optimised for SPARSE
mixture detection. For `n` p-values sorted `p_(1) ≤ p_(2) ≤ …
≤ p_(n)`, form

```
HC_i = √n · (i/n − p_(i)) / √(p_(i) · (1 − p_(i)))
HC*  = max_{α₀ ≤ i/n ≤ α_end} HC_i
```

Attains the OPTIMAL detection boundary in Ingster's (1997)
sparse-signal regime: mixture detection succeeds asymptotically
when signal strength exceeds `√(2 log n)` in a specific
sparsity-signal-strength trade-off. Foundational for genomics
and cosmology signal-detection.

## Files

- `python/higher_criticism_multiple_testing.py` — Three
  regimes with n=1 000 z-scores. Frac signal 1% at μ=3.5:
  HC* = 11.2. Frac 5% at μ=3.0: HC* = 38.1. Even the very
  sparse 0.1% at μ=5.0 gives HC* = 2.07 > threshold 1.97.
  All three globally reject the null.
- `r/higher_criticism_multiple_testing.R` — no dominant R
  package; custom implementation is standard (R);
  `statsmodels.stats.multitest`, from-scratch (Python).

## When to use

- **Sparse signal detection** — GWAS, brain imaging, cosmic
  ray physics, spam / anomaly detection.
- **Global-null test alternative to Fisher/Stouffer** — when
  a small fraction of tests carry the signal.
- **Feature-screening in high-dim regression** — HC-thresholded
  variable selection.

## When NOT to use

- **Dense-alternative regime** — many small deviations; use
  Fisher combined p or aggregate z / Stouffer.
- **When correlated tests** — HC's null distribution assumes
  independence; use permutation p-values or correlated-HC
  extensions.
- **When effect sizes are the target** — HC is a global
  detector, not an estimator.

## Assumptions & caveats

- **α₀ tuning** — usually 0.005 lower and 0.5 upper for the
  sup range; too wide includes noisy tails.
- **Null threshold** — asymptotically `√(2 log log n)` under
  independence; for finite n use simulation.
- **Adjusted HC / HC+** — Donoho-Jin (2008) refinements
  reduce finite-sample bias; use for very small n.
- **Correlated p-values** — reduces effective sample size;
  use permutation resampling to calibrate.
- **Multiple testing** — HC is a GLOBAL test; for individual-
  hypothesis rejection combine with BH-FDR.

## Related in this repo

- `multiple-testing-corrections`, `multiple-metrics-fdr`,
  `benjamini-hochberg` (in `multiple-testing-corrections`),
  `storey-q-value` (if present) — FDR neighbours.
- `fisher-combine-pvalues`, `mmd-two-sample-test`,
  `permutation-tests` — global-test alternatives.
- `sure-independence-screening`, `stability-selection` —
  feature-screening cousins.
- `gwas`, `gsea`, `differential-expression` — genomic
  application domains.

## Run

```
python techniques/higher-criticism-multiple-testing/python/higher_criticism_multiple_testing.py
Rscript techniques/higher-criticism-multiple-testing/r/higher_criticism_multiple_testing.R
```

**Refs:** Donoho, D. and Jin, J. "Higher criticism for detecting sparse heterogeneous mixtures." *Ann. Statist.*, 32: 962-994, 2004; Ingster, Y.I. "Some problems of hypothesis testing leading to infinitely divisible distributions." *Math. Methods Statist.*, 6: 47-69, 1997.

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
