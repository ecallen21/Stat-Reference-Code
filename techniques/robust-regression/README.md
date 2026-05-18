# Robust Regression (Reference §5.11)

OLS minimizes `Σeᵢ²` — squared loss grows without bound, so a single bad point can dominate the fit. Robust methods replace the squared loss with something that grows more slowly (or stops growing) in the tail.

## Huber M-estimator

`ρ(r) = ½ r²` for `|r| ≤ k`,  `k|r| − ½k²` for `|r| > k`.

Minimizes `Σ ρ(eᵢ / σ̂)`. Standard tuning constant `k = 1.345` gives ~95% efficiency at the normal — almost no penalty when assumptions hold, but bounded influence from a single point.

**Algorithm — IRLS (iteratively reweighted least squares):**
1. OLS to get initial β.
2. Compute residuals; estimate scale σ̂ via MAD.
3. Weights: `wᵢ = 1` if `|uᵢ| ≤ k`, else `k/|uᵢ|` (where `uᵢ = eᵢ/σ̂`).
4. WLS step: `β ← (X'WX)⁻¹X'Wy`.
5. Repeat to convergence.

## Higher-breakdown options (mentioned, not implemented from scratch)

| Method | Breakdown | Efficiency at normal | Notes |
|--------|-----------|---------------------|-------|
| OLS | 0% | 100% | One bad point can break it |
| Huber M | 0%* | ~95% | Bounded influence on small residuals, not on `xᵢ` |
| **MM-estimator** (Yohai 1987) | **50%** | ~95% | S-estimator init + M-step; **modern default** when robustness matters |
| **LTS** (Least Trimmed Squares) | **50%** | ~7% (then refit) | Minimize the `h` smallest squared residuals; very robust, low efficiency without a refit step |
| LAD (`L1`) | 0%* | ~64% | Minimize `Σ|eᵢ|`; lower efficiency than Huber |

\*Breakdown of an M-estimator is 0% **if** observations have high leverage in `x` — Huber bounds the influence of large residuals but not large `x`. MM and LTS handle leverage points; that's the practical reason to prefer them for heavily contaminated data.

For production use, prefer:
- **R**: `robustbase::lmrob` (MM-estimator) or `MASS::rlm` (Huber).
- **Python**: `statsmodels.RLM` (Huber / Tukey / etc.); MM-estimators not in stock statsmodels — see `scikit-learn`'s `HuberRegressor` for a related estimator, or the `robust` package.

## When to use it

- Visible outliers / heavy tails in residuals.
- Mixed populations (some "normal" + some "contamination").
- Sensitivity analysis: "do my OLS conclusions survive if I downweight obvious outliers?"

## When NOT to use it (without thought)

- If the "outliers" are genuine signal (the most informative data points), robust regression hides them. **Investigate first.**
- For small `n`, robust methods have low power.
- For inference, the SEs from a robust fit are asymptotic approximations; bootstrap if you need tight CIs.

## Files
- `python/robust_regression.py` — from-scratch Huber M-estimator via IRLS with MAD scale; demos a `y = 1 + 2x + ε` fit corrupted by two extreme points (OLS slope = 1.69, Huber slope = 2.03, true = 2.00). Compares against `statsmodels.RLM(M=HuberT())`.
- `r/robust_regression.R` — from-scratch + `MASS::rlm` (Huber) and `robustbase::lmrob` (MM-estimator) library cross-checks; notes on `robustbase::ltsReg`.
- PySpark: N/A — these are residual-IRLS loops on small-to-medium `n`. For large data with outliers, sample, fit robustly on the sample, and apply.

## Run
```
python techniques/robust-regression/python/robust_regression.py
Rscript techniques/robust-regression/r/robust_regression.R
```

**Refs:** Huber, "Robust Estimation of a Location Parameter," *Ann. Math. Stat.* 35(1), 73–101, 1964; Yohai, "High Breakdown-Point and High Efficiency Robust Estimates for Regression," *Ann. Stat.* 15(2), 642–656, 1987; Rousseeuw & Leroy, *Robust Regression and Outlier Detection*, Wiley, 1987.

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
