# Censored Quantile Regression (Reference §33.3)

Powell (1986) extended quantile regression to **censored outcomes** by
projecting the linear predictor through the censoring threshold:

```
min_β  Σ ρ_τ( y_i − min( x_iᵀ β, C ) )        (right-censoring at C)
min_β  Σ ρ_τ( y_i − max( x_iᵀ β, L ) )        (left-censoring at L)
```

Naive QR on censored `y` is **biased** because the loss ignores that
observations at the ceiling are lower bounds on the latent variable.
Powell's CQR is **consistent** under mild conditions.

## When to use

- **Top-coded / bottom-coded survey outcomes** — income capped at
  `$500 k+`, hospital LOS capped at 30 days.
- **Type-I censored survival times** with fixed follow-up.
- **Quantile-of-interest models** where censoring makes Cox / AFT
  awkward.

## When NOT to use

- **Random / informative censoring** — need Portnoy 2003 (recursive
  reweighting) or Peng-Huang 2008 (martingale).
- **Very heavy censoring above the quantile of interest** — the
  quantile is not identified.

## Files

- `python/censored_quantile_regression.py` — from-scratch Powell CQR
  via subgradient descent with active-set masking at the ceiling.
  Demo on right-censored data (21 % censored above `C = 3`) with true
  `β = (1.0, 1.5)`. Naive QR gives biased slope 0.93-1.13; Powell CQR
  recovers slope ≈ 1.45 across `τ ∈ {0.25, 0.50, 0.75}`.
- `r/censored_quantile_regression.R` — `quantreg::crq` (Powell + Portnoy
  2003); Python via `reticulate`.

## Assumptions & caveats

- **Objective is non-convex** — Powell's original paper used iterative
  LP (BRCENS); Portnoy 2003 gives a more robust recursion; the demo's
  subgradient solver is illustrative, not production.
- **Censoring point must be known** — random censoring needs weight
  adjustment (Portnoy / Peng-Huang).
- **Quantile below the censoring proportion** may be well-identified
  even with heavy censoring, but quantiles far above the ceiling
  cannot be recovered.
- **Standard errors** need block-bootstrap for Powell; asymptotic
  sandwich available under regularity conditions.

## Related in this repo

- `quantile-regression` — the uncensored parent.
- `bayesian-quantile-regression`, `additive-quantile-regression`,
  `expectile-regression` — sibling extensions.
- `cox-ph`, `aft-survival` — survival-time alternatives when
  the target is time-to-event.
- `tobit-regression` (if present) — mean-based analogue for censored
  outcomes.

## Run

```
python techniques/censored-quantile-regression/python/censored_quantile_regression.py
Rscript techniques/censored-quantile-regression/r/censored_quantile_regression.R
```

**Refs:** Powell, J.L. "Censored regression quantiles." *Journal of Econometrics*, 1986; Portnoy, S. "Censored regression quantiles." *JASA*, 2003; Peng, L. & Huang, Y. "Survival analysis with quantile regression models." *JASA*, 2008.

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
