# Safe Screening for LASSO (Reference §47.336)

El Ghaoui, Viallon & Rabbani (2010); Fercoq et al (2015).
Rules that DEFINITIVELY exclude features from the active set
BEFORE running the LASSO solver:

```
SAFE rule (dual bound):
    |x_jᵀ y| < λ − r · ‖x_j‖₂   =>   β_j = 0
```

Reduces problem size from `d` to few active features; used
inside LASSO / SVM solvers (celer, glmnet) for O(d) speed-ups
at large `d`.

## Files

- `python/safe_screening_lasso.py` — SAFE screening + basic
  coordinate-descent solver. Demo n=200, d=500 with 8 true
  nonzeros. Screening keeps a small fraction of columns; full
  and screened solutions match to numerical zero.
- `r/safe_screening_lasso.R` — `glmnet` (strong-rule),
  `biglasso` (R); `celer.Lasso`, `sklearn.Lasso`, from-scratch
  (Python).

## When to use

- **Very high-dim LASSO** (d ≫ n) — screening exclusion
  makes the solver linear in the ACTIVE set.
- **Path solving over a λ grid** — dynamic screening reuses
  bounds across neighbouring λ.
- **Elastic net / logistic-LASSO** — similar rules apply.

## When NOT to use

- **Small d** — screening overhead outweighs the saving.
- **Non-convex penalties (SCAD, MCP)** — safe rules require
  duality; adapt to strong-rule heuristics.
- **When λ is very small** — few features screened out.

## Assumptions & caveats

- **Screening is EXACT** — features excluded are provably
  zero at the LASSO optimum.
- **Bound tightness** — vanilla El Ghaoui bound is loose;
  GAP-safe rules (Fercoq 2015) are much tighter.
- **Warm-start along λ path** — critical for dynamic
  screening.
- **Strong-rule vs safe** — strong-rule is heuristic (may
  err); safe never excludes an active feature.

## Related in this repo

- `proximal-gradient-method`, `fista-accelerated-proximal`,
  `proximal-newton` — companion solvers.
- `adaptive-lasso`, `fused-lasso`, `group-lasso`,
  `scad-mcp-penalties`, `debiased-lasso`, `stability-selection`,
  `sure-independence-screening` — related LASSO family.

## Run

```
python techniques/safe-screening-lasso/python/safe_screening_lasso.py
Rscript techniques/safe-screening-lasso/r/safe_screening_lasso.R
```

**Refs:** El Ghaoui, L., Viallon, V. and Rabbani, T. "Safe feature elimination for the LASSO and sparse supervised learning problems." *arXiv:1009.4219*, 2010; Fercoq, O., Gramfort, A. and Salmon, J. "Mind the duality gap: safer rules for the Lasso." In *ICML*, 2015.

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
