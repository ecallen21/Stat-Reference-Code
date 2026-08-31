# Fused LASSO / Total Variation (Reference §32.13)

Tibshirani et al. (2005). Adds a penalty on **adjacent differences** of
coefficients:

```
min_β  ½ ‖ y − X β ‖²  +  λ₁ Σ_j |β_j|  +  λ₂ Σ_j |β_j − β_{j-1}|
```

For 1-D signal denoising (`X = I`), fused LASSO reduces to
**total-variation (TV) denoising** — preserves piecewise-constant
structure while removing noise. Standard in image denoising,
changepoint detection, comparative genomic hybridisation (CGH),
copy-number variation.

## When to use

- **Piecewise-constant regression** — CGH, CNV, changepoint problems.
- **Smooth-with-jumps signals** — image edges, trend breaks.
- **1-D or graph-structured coefficients** — ordering matters.

## When NOT to use

- **Smoothly-varying coefficients** — use ridge / spline penalties.
- **Unordered features** — the TV penalty is meaningless.
- **Very high-dim with weak sparsity** — coordinate-descent for fused
  LASSO is slower than plain LASSO.

## Files

- `python/fused_lasso.py` — from-scratch **Chambolle 2004 dual
  projection** for 1-D TV denoising. Demo on a 120-point signal with
  3 plateaux + Gaussian noise. Sweep `λ ∈ {0.05, 0.15, 0.40}`:
  **TV_MSE drops 0.083 → 0.017** at `λ = 0.40` (near-perfect
  recovery); estimated changepoints 46 → 5 (true = 2).
- `r/fused_lasso.R` — `genlasso::fusedlasso1d`, `genlasso::trendfilter`
  (R); `skimage.restoration.denoise_tv_chambolle`, `cvxpy`,
  `ruptures` (Python).

## Assumptions & caveats

- **Chambolle dual** — the demo's iterations are Chambolle-Pock; only
  approximately piecewise constant. For exact plateaux, use
  `genlasso::fusedlasso1d` or dynamic-programming solvers (PELT).
- **λ tuning** — cross-validation on held-out subsequences.
- **2-D / graph TV** — the Chambolle-Pock scheme generalises to any
  discrete-gradient operator; see graph fused LASSO (Xin 2015).
- **Trend filtering** (Kim-Koh-Boyd 2009 / Tibshirani 2014) is the
  higher-order sibling — piecewise-polynomial fits.
- **Interaction with fused-sparse LASSO** — combines TV with an L1
  penalty on the levels themselves for sparse-plateaux problems.

## Related in this repo

- `ridge-lasso-elasticnet` — the L1 parent.
- `changepoint-detection` (if present) — the same problem, DP solver.
- `wavelet-analysis`, `spectral-analysis` — frequency-domain
  denoising alternatives.
- `kernel-density-estimation`, `local-regression-loess` — smoothing
  siblings.
- `time-series-*` — trends with breaks appear often in financial /
  hydrology series.

## Run

```
python techniques/fused-lasso/python/fused_lasso.py
Rscript techniques/fused-lasso/r/fused_lasso.R
```

**Refs:** Tibshirani, R., Saunders, M., Rosset, S., Zhu, J. & Knight, K. "Sparsity and smoothness via the fused LASSO." *JRSS-B*, 2005; Chambolle, A. "An algorithm for total variation minimization and applications." *Journal of Mathematical Imaging and Vision*, 2004; Tibshirani, R.J. "Adaptive piecewise polynomial estimation via trend filtering." *Annals of Statistics*, 2014.

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
