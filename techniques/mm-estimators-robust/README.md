# MM-Estimators for Robust Regression (Reference §17.x extra)

Two-stage estimator (Yohai 1987) combining **high breakdown** with **high
efficiency**:

- **Stage 1 — S-estimator**: Tukey biweight ρ with tuning `c₁ ≈ 1.548`, δ = 0.5. Achieves 50% breakdown-point resistance to outliers; the M-scale of residuals is minimised subject to `(1/n) Σ ρ(r_i / σ) = δ`.
- **Stage 2 — M-estimator**: Tukey biweight with a **wider** tuning constant `c₂` (95% efficiency at Gaussian at c=4.685; ~85% at c=3.44 with better outlier resistance in practice). IRLS from the Stage-1 fit `β_S`, holding scale `σ = σ_S` fixed.

The result inherits the S-estimator's breakdown and the M-estimator's efficiency at the Gaussian model.

## Why a good starting point matters

Naive IRLS from an OLS start under high-leverage contamination collapses back
to the OLS fit — the outliers pull the fit toward themselves, they don't get
down-weighted, and the M-step is trapped in a bad local optimum. The
subset-search S-estimator (FAST-S; Rousseeuw-Van Driessen 1999) breaks that
by trying many random `p`-subsets and picking the one whose full-sample
M-scale is smallest.

## When to use

- **Any regression with suspected outliers or leverage points** — before or as a companion to OLS.
- **Diagnostic-driven robustness** — if OLS residuals show heavy tails or a few large points, MM gives an alternative fit.
- **Standard errors sensitivity** — combine with `sandwich-robust-se` for heteroskedasticity + `mm-estimators-robust` for outliers.

## Files

- `python/mm_estimators_robust.py` — FAST-S-style subset-search initialisation, S-IRLS refinement, then M-step at c=3.44. Demo (n=200, 20% high-leverage outliers with y+15 and x+4 shift, true β = [1, 2, −1]): OLS = [1.55, 3.21, −1.04] (badly biased), MM = [1.03, 1.97, −0.89] (nails the truth), statsmodels TukeyBiweight from OLS start collapses to OLS-like [1.54, 3.20, −1.05] because it lacks the S-init.
- `r/mm_estimators_robust.R` — `robustbase::lmrob` (recommended default; uses FAST-S + M-step + Koller-Stahel defaults).

## Related methods

- **M-estimators** — Huber, biweight; robust to outliers but 0% breakdown against high-leverage points (see `quantile-regression` for a distributionally-robust alternative).
- **S-estimators** — 50% breakdown; low efficiency (~28% for biweight at c=1.548). Rarely reported alone.
- **LTS / LMS** (least trimmed / median squared residuals) — 50% breakdown; even less efficient than S; used as starting point for MM.
- **REWLSE / adaptive MM** — cross-validated tuning of c₂.
- **Robust GLMs** — `robustbase::glmrob` for Poisson / logistic robust fits.

## Assumptions & caveats

- **Efficiency vs resistance trade-off**: c₂ = 4.685 → 95% efficient but sensitive to high-leverage outliers; c₂ = 3.44 → 85% efficient, much more resistant. Report both.
- **Non-Gaussian noise** — MM is designed for Gaussian errors with outlier contamination; a heavy-tailed but symmetric noise distribution may not need MM.
- **Contamination fraction > 50%** breaks the S-estimator; use LMS (breakdown = ⌊(n − p) / 2 + 1⌋ / n) as init or accept the estimator's limits.
- **Convergence** — MM can have multiple local optima; report `beta_S`, `beta_MM`, and OLS side-by-side.
- **Standard errors** are asymptotic; use the bootstrap for finite-sample inference on unusual data.
- **Random-subset initialisation** makes the result stochastic; increase `n_subsets` and fix a seed for reproducibility.

## Run

```
python techniques/mm-estimators-robust/python/mm_estimators_robust.py
Rscript techniques/mm-estimators-robust/r/mm_estimators_robust.R
```

**Refs:** Yohai, V.J. "High breakdown-point and high efficiency estimates for regression." *Ann. Statist.* 15(2), 642–656, 1987; Rousseeuw, P.J. & Van Driessen, K. "Computing LTS regression for large data sets." *Data Min. Knowl. Discov.* 12(1), 29–45, 2006; Koller, M. & Stahel, W.A. "Sharpening Wald-type inference in robust regression for small samples." *Comput. Stat. Data Anal.* 55(8), 2504–2515, 2011.

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
