# Quantile Regression (Reference §5.15)

OLS estimates the **conditional mean** `E[y | X]`. **Quantile regression** (Koenker & Bassett 1978) estimates the **conditional quantile** `Q_τ(y | X)` at any chosen `τ ∈ (0, 1)`:

```
minimize  Σ_i ρ_τ(y_i − X_i β)
    ρ_τ(u) = u · (τ − I(u < 0))
           = max(τ u, (τ − 1) u)          (pinball / check loss)
```

- `τ = 0.5` → conditional **median** (L1 regression; robust to outliers).
- `τ = 0.1, 0.9` → tails; reveal heterogeneous effects.

## Solvers

- **Linear programming** (Koenker's original) — exact; `scipy.optimize.linprog(method="highs")` handles moderate `n`.
- **Smoothed pinball + BFGS** — simpler to implement; slight bias for `τ` far from 0.5.

## Reading a quantile-regression fit

Fit at a grid of `τ` (0.1, 0.25, 0.5, 0.75, 0.9). Plot each slope as a function of `τ`:

- **Slopes constant in `τ`** — pure location shift; OLS is fine.
- **Slopes fan out** — heteroscedastic effect; `x` widens the outcome distribution.
- **Slopes cross** — genuine distributional differences; only visible in QR.

## Files

- `python/quantile_regression.py` — smoothed pinball BFGS + optional LP fallback. Demo (n = 500, heteroscedastic errors that grow with x): slopes fan from 1.31 (τ = 0.1) to 2.58 (τ = 0.9); matches `statsmodels.regression.quantile_regression.QuantReg` to two decimals.
- `r/quantile_regression.R` — `quantreg::rq` (Koenker's canonical R implementation, supports many solvers and confidence bands).

## When to use

- **Skewed / heavy-tailed** outcomes where the mean is a poor summary (income, health-care expenditure).
- **Heterogeneous effects** — does `x` mainly move the upper or lower tail?
- **Robust regression** at `τ = 0.5` (median regression).
- **Reference bands** — growth curves showing 3rd, 50th, 97th percentile as a function of age.

## Assumptions & caveats

- **Distribution-free** — no error-distribution assumption; just needs enough observations near the target quantile.
- **Standard errors**: robust ("nid") sandwich SEs are standard; bootstrap SEs are safer under heteroscedasticity.
- **Sample size**: extreme quantiles (τ ≤ 0.05 or ≥ 0.95) need large n; consider extremal-quantile methods (Chernozhukov 2005) for the very tails.
- **Non-crossing constraints**: raw QR at multiple τ can produce crossing curves in finite samples; use `quantreg::rq(method = "fnb")` or non-crossing quantile methods.

## Run

```
python techniques/quantile-regression/python/quantile_regression.py
Rscript techniques/quantile-regression/r/quantile_regression.R
```

**Refs:** Koenker, R. & Bassett, G. "Regression quantiles." *Econometrica* 46(1), 33–50, 1978; Koenker, R. *Quantile Regression*, Cambridge, 2005.

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
