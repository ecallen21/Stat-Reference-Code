# Coarsened Exact Matching (Reference §15.10)

Iacus-King-Porro (2012). Discretise ("coarsen") continuous
covariates into bins, then **exact match** treated to control units
within each coarsened cell. Compute weights so within-cell controls
match treated counts.

## Why CEM

- **Non-parametric** — no PS model to misspecify.
- **Guaranteed balance** on the coarsened scale.
- **L₁ imbalance** provides a single-number diagnostic.
- **Monotonic imbalance bounding** — coarser bins → looser
  balance guarantee; the analyst chooses the trade-off.

## When to use

- **Modest-dim, mostly-categorical** confounders where bins map
  naturally to strata.
- **Reviewer-friendly** — the matching cells are inspectable.

## When NOT to use

- **High-dim continuous confounders** — most cells empty of
  matches.
- **Curse of dimensionality** — 10 bins × 10 covariates → 10¹⁰
  cells.

## Files

- `python/coarsened_exact_matching.py` — bin + exact-match +
  Hájek-style weight (custom). Demo (n=2000, 2 confounders,
  quartile bins, true ATT=0.5): **945 / 957 treated retained,
  1033 / 1043 controls**; naive difference **+0.94 (biased)** →
  **CEM ATT +0.59** — close to truth.
- `r/coarsened_exact_matching.R` — `MatchIt::matchit(method='cem')`,
  `cem`, `cobalt` (R); custom (Python).

## Assumptions & caveats

- **Bin choice matters** — too coarse → residual imbalance; too
  fine → many unmatched cells.
- **Unmatched units dropped** — reduces effective sample; report
  the drop rate.
- **Estimand is ATT** on the matched subsample (or ATE with two-
  sided weighting).
- **Combine with regression** (double-robust CEM) to adjust for
  within-cell residual imbalance.

## Related in this repo

- `propensity-score-matching`, `iptw`, `entropy-balancing` —
  companion adjustment methods.
- `mahalanobis-distance-matching` (if present) — continuous
  alternative.

## Run

```
python techniques/coarsened-exact-matching/python/coarsened_exact_matching.py
Rscript techniques/coarsened-exact-matching/r/coarsened_exact_matching.R
```

**Refs:** Iacus, S.M., King, G., & Porro, G. "Causal inference without balance checking: coarsened exact matching." *Political Analysis*, 2012.

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
