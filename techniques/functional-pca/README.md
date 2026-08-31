# Functional PCA (Reference §31.2)

Ramsay & Silverman (2005). Each observation is a **curve** `x_i(t)`
over a common domain. FPCA finds orthonormal basis functions
`φ_k(t)` and per-curve scores `ξ_ik` such that

```
x_i(t)  ≈  μ(t)  +  Σ_k ξ_ik · φ_k(t).
```

## Two variants

- **Discrete FPCA** — evaluate curves on a common grid, run PCA on the
  matrix of function values.
- **Basis-expanded FPCA** — expand each curve in a spline basis, run
  PCA on the coefficient vectors (Silverman 1996 penalised variant).

## When to use

- **Growth curves, spectroscopy, wearable-sensor day-profiles** —
  any smooth curve as unit of observation.
- **Dimension reduction for functional regression** — feed FPC scores
  into a downstream model.
- **Interpretable modes of variation** — first FPC often has a
  natural physical interpretation.

## When NOT to use

- **Very few, very sparse observations per curve** — PACE (§31.8) or
  penalised methods better.
- **Multivariate curves** (multivariate FDA) — extend via joint FPCA.
- **Non-smooth / spike-heavy signals** — wavelet-based DR alternative.

## Files

- `python/functional_pca.py` — from-scratch discrete FPCA via SVD.
  Demo on 50 shifted sinusoids with hidden `amplitude` and
  `phase-shift` factors: **PC1 corr 0.97 with phase, PC2 corr 0.92
  with amplitude**; first two PCs together explain 98 % of variance.
- `r/functional_pca.R` — `fda::pca.fd`, `fda.usc`, `refund` (R);
  `scikit-fda`, `fdasrsf` (Python).

## Assumptions & caveats

- **Grid alignment** — discrete FPCA assumes common time points; use
  PACE for sparse / irregular curves.
- **Number of components** — scree plot / cumulative variance /
  cross-validation.
- **Smoothing before PCA** — reduces noise; penalised FPCA
  incorporates it into the objective.
- **Phase vs amplitude confusion** — heavy phase misalignment is
  better handled with `curve-registration` first.
- **Interpretation of `φ_k`** — plot alongside `μ ± c σ_k φ_k` to
  visualise the mode.

## Related in this repo

- `functional-regression`, `functional-anova`, `functional-clustering`
  — the downstream FDA family (this batch).
- `curve-registration` — sibling for phase alignment (this batch).
- `pca` (if present) / `sparse-pca` / `kernel-pca` — multivariate
  cousins.
- `time-series-*` — related when curves are time series.

## Run

```
python techniques/functional-pca/python/functional_pca.py
Rscript techniques/functional-pca/r/functional_pca.R
```

**Refs:** Ramsay, J.O. & Silverman, B.W. *Functional Data Analysis*, Springer, 2005 (Ch. 8); Silverman, B.W. "Smoothed functional principal components analysis by choice of norm." *Annals of Statistics*, 1996.

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
