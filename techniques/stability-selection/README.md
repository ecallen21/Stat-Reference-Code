# Stability Selection (Reference §32.6)

Meinshausen & Bühlmann (2010) — a **model-agnostic wrapper** that
converts any selection procedure (LASSO, RF importance, boosting) into
one with FDR-like error control.

## Algorithm

Repeatedly fit the base selector on random subsamples of size `n / 2`;
for each feature track the **selection frequency** across subsamples.
Keep features whose frequency exceeds a threshold `π_thr` (typically
0.6-0.9).

Under exchangeability + a symmetry condition the **expected number of
false positives** is bounded:

```
𝔼[V]  ≤  q² / ( (2 π_thr − 1) · p )
```

where `q` = average number of variables selected per subsample and
`p` = total feature count.

## When to use

- **Model-agnostic FDR control** — works with LASSO, RF, boosting.
- **Post-LASSO stabilisation** — smooths out the LASSO's sensitivity
  to `λ`.
- **Discovery-oriented analysis** — genomic screens, chemical
  screening.

## When NOT to use

- **Very small n** — subsampling to `n / 2` degrades power.
- **Highly correlated features** — the selection frequency
  distributes across correlated groups (see cluster-stability
  extensions).
- **You have a formal FDR method** — knockoffs / Benjamini-Hochberg
  give tighter guarantees.

## Files

- `python/stability_selection.py` — from-scratch coordinate-descent
  LASSO + subsampling loop (`B = 60`, subsample = 0.5) across a
  `λ`-grid. Demo `n=200, d=30`, 3 signals: **all 3 true signals hit
  max-frequency 1.00**; one false positive at 0.93. At `π_thr ∈
  {0.60, 0.75, 0.90}` the selection is `TP = 3, FP = 1`.
- `r/stability_selection.R` — `stabs` (R reference);
  `stability-selection` (Python).

## Assumptions & caveats

- **Symmetry condition** — required for the FDR bound; approximately
  holds for LASSO.
- **π_thr** — larger threshold = fewer false positives, less power.
  Conservative default 0.75.
- **`λ`-grid** — take the max frequency across `λ`s; this is the
  "stability path".
- **Complementary-pairs variant** (Shah-Samworth 2013) gives a tighter
  bound.
- **Correlated features** — a group of collinear predictors shares
  selection frequency; consider group-LASSO base selector.

## Related in this repo

- `ridge-lasso-elasticnet`, `adaptive-lasso`, `scad-mcp-penalties` —
  the base selectors this wraps.
- `model-x-knockoffs`, `debiased-lasso` — FDR / CI alternatives.
- `bootstrap`, `bca-bootstrap`, `jackknife` — resampling cousins.
- `cross-validation` — the tuning framework often paired with SS.

## Run

```
python techniques/stability-selection/python/stability_selection.py
Rscript techniques/stability-selection/r/stability_selection.R
```

**Refs:** Meinshausen, N. & Bühlmann, P. "Stability selection." *JRSS-B*, 2010; Shah, R.D. & Samworth, R.J. "Variable selection with error control: another look at stability selection." *JRSS-B*, 2013.

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
