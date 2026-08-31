# Expectile Regression (Reference §33 — Semiparametric / Distribution-Free)

Newey & Powell (1987) — the **asymmetric-squared-loss** analogue of
quantile regression. Instead of the check loss `ρ_τ(u) = u(τ − 𝟙[u<0])`,
expectile regression uses the asymmetric squared loss:

```
min_β  Σ_i  w_τ(y_i − x_iᵀ β) · (y_i − x_iᵀ β)²
   w_τ(u) = τ     if u > 0
           1 − τ  if u ≤ 0.
```

Solved by iteratively-reweighted least squares (IRLS).

## Expectile vs quantile

| Property                | Quantile (τ) | Expectile (τ)          |
|-------------------------|--------------|------------------------|
| Loss                    | check (L1-ish) | asymmetric squared    |
| Smoothness              | non-differentiable | smooth              |
| Coherent risk measure   | not exact    | yes (Bellini 2014)     |
| Invariant to monotone f | yes          | no                     |
| Sensitivity to spread   | insensitive  | sensitive              |
| Standard errors         | asymptotic sandwich | closed-form sandwich |

- **`τ = 0.5`** ⇒ ordinary least squares (mean).
- Expectiles are more sensitive to distribution *spread* than
  quantiles; useful when tail-magnitude, not tail-probability,
  matters (risk management, extreme weather).

## When to use

- **Risk / actuarial** — expectiles are coherent risk measures.
- **Smooth optimisation** — a differentiable analogue of QR.
- **Fast standard errors** — closed-form sandwich available.

## When NOT to use

- **Interpretation as probabilities** — quantiles are direct
  probability statements; expectiles are not.
- **Non-invariant transformations** — expectiles change with monotone
  transforms of `y`; quantiles do not.
- **Robustness to outliers** — squared loss is not robust; use QR.

## Files

- `python/expectile_regression.py` — from-scratch IRLS expectile
  regression. Demo on heteroscedastic linear data at `τ ∈ {0.10, 0.25,
  0.50, 0.75, 0.90}`. **τ = 0.5 exactly reproduces OLS** (intercept
  0.978, slope 0.449); intercepts spread 0.14 → 1.86 across expectile
  levels, reflecting the growing conditional spread.
- `r/expectile_regression.R` — `expectreg` (R); `statsmodels`,
  `HuberRegressor` (adjacent, Python).

## Assumptions & caveats

- **Not invariant to monotone transformation of y** — a common
  interpretive trap.
- **IRLS convergence** — usually fast (< 20 iterations); may oscillate
  near boundaries; add small ridge for stability.
- **Standard errors** — sandwich SEs from Newey-Powell's asymptotics;
  bootstrap for small n.
- **Nonlinear extension** — additive expectiles via P-spline bases
  (`expectreg`).
- **Multiple expectiles** — crossings are less common than for
  quantiles because the squared loss smooths the fit.

## Related in this repo

- `quantile-regression` — the median-based cousin.
- `additive-quantile-regression`, `bayesian-quantile-regression`,
  `censored-quantile-regression` — quantile-family extensions.
- `distributional-regression`, `gamlss` — full-distribution
  alternatives.
- `robust-regression` — different asymmetric-loss family (Huber).

## Run

```
python techniques/expectile-regression/python/expectile_regression.py
Rscript techniques/expectile-regression/r/expectile_regression.R
```

**Refs:** Newey, W.K. & Powell, J.L. "Asymmetric least squares estimation and testing." *Econometrica*, 1987; Bellini, F. et al. "Generalized quantiles as risk measures." *Insurance: Mathematics and Economics*, 2014.

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
