# Additive Quantile Regression (Reference §33.13)

Extends quantile regression with **smooth nonlinear effects** via basis
expansions (natural cubic splines, P-splines, thin-plate) and a check-
loss objective:

```
min_β  Σ_i ρ_τ( y_i − Σ_j g_j(x_ij) )   +   λ · Σ_j ‖β_j‖²
```

Each `g_j` is expanded in a spline basis; the loss is convex and
LP-friendly.

## When to use

- **Nonlinear conditional quantiles** — dose response, growth curves,
  climate extremes.
- **Heteroscedasticity** — different quantiles have different shapes,
  and additive QR fits them directly.
- **Same interpretability as QR** with the flexibility of GAMs.

## When NOT to use

- **Very small n** — spline df eats degrees of freedom.
- **Extremes (τ near 0 or 1)** — tail quantiles are noisy; consider
  extreme-value QR or conformal-quantile methods.
- **Multiple covariates with interactions** — tensor-product splines
  are heavy; boosted trees may fit better.

## Files

- `python/additive_quantile_regression.py` — from-scratch cubic
  truncated-power B-spline basis + standardised subgradient descent
  on the check loss at `τ ∈ {0.10, 0.50, 0.90}`. Demo on noisy
  `sin(1.5x)` with heteroscedastic variance: **q50 tracks sin(1.5x)
  reasonably**; q10-q90 spread grows from ~1.0 at `x = 0` to ~2.7
  at `x = ±1`, consistent with the increasing conditional scale.
- `r/additive_quantile_regression.R` — `quantreg::rqss`, `qgam` (R);
  `statsmodels`, `pyGAM`, `scikit-learn QuantileRegressor` (Python).

## Assumptions & caveats

- **Quantile crossing** — separate fits per τ can cross; use joint
  monotone-quantile fits (Bondell 2010; qgam) or `distributional-
  regression` if it matters.
- **Bandwidth / knot placement** — cross-validation on check loss
  is standard; the demo uses 10 knots uniformly.
- **Standardise basis columns** — subgradient descent is very
  sensitive to feature scale.
- **Feature interactions** — tensor-product bases blow up quickly;
  consider boosting for interactions.
- **Standard errors** — `qgam` and `rqss` use asymptotic sandwich;
  bootstrap is safer for small n.

## Related in this repo

- `quantile-regression` — the linear parent.
- `bayesian-quantile-regression`, `censored-quantile-regression`,
  `expectile-regression` — sibling extensions.
- `gamlss`, `distributional-regression` — full-distribution alternatives.
- `varying-coefficient-model` — smooth coefficients along a modifier.
- `restricted-cubic-splines` (if present) — same spline family for
  mean regression.

## Run

```
python techniques/additive-quantile-regression/python/additive_quantile_regression.py
Rscript techniques/additive-quantile-regression/r/additive_quantile_regression.R
```

**Refs:** Koenker, R. *Quantile Regression*, Cambridge University Press, 2005 (Ch 6); Fasiolo, M., Wood, S.N., Zaffran, M., Nédellec, R. & Goude, Y. "Fast calibrated additive quantile regression (qgam)." *JASA*, 2021; Bondell, H.D., Reich, B.J. & Wang, H. "Noncrossing quantile regression curve estimation." *Biometrika*, 2010.

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
