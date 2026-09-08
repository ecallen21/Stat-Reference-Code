# RIF Unconditional Quantile Regression (Reference §47.46)

Firpo, Fortin & Lemieux (2009). Standard (Koenker-Bassett)
quantile regression models the CONDITIONAL quantile Q_{Y|X}(τ|x).
RIF-OLS instead targets the change in the **marginal (unconditional)**
quantile of Y induced by shifting x.

Recentered Influence Function

    RIF(y; q_τ, F_Y) = q_τ + (τ − 𝟙{y ≤ q_τ}) / f_Y(q_τ)

Running OLS on the RIF gives per-covariate **unconditional partial
effects** — used widely for wage-decomposition (Firpo-Fortin-Lemieux
2018) and inequality analysis.

## Files

- `python/rif_regression_firpo.py` — RIF-OLS with density
  estimated by `gaussian_kde`; comparison to Koenker-Bassett
  conditional quantile regression via LP surrogate. Demo (n=4000,
  heteroskedastic y with binary moderator x₂):
  - τ=0.10: RIF x₂ effect +0.30, cond QR x₂ +0.17
  - τ=0.50: RIF x₂ effect +0.52, cond QR x₂ +0.49
  - τ=0.90: RIF x₂ effect +0.75, cond QR x₂ +0.85
    (differ where the conditional distribution shape depends on x).
- `r/rif_regression_firpo.R` — `dineq::rifr`, `uqr` (R);
  `statsmodels` + custom, from-scratch (Python).

## When to use

- **Marginal quantile change** — how does the 10th percentile of
  wages change with education?
- **Wage / income decomposition** — Firpo-Fortin-Lemieux 2018.
- **Health-inequality analysis** — RIF at each quantile.
- **Policy-relevant unconditional effects** — for shifts in a
  covariate distribution.

## When NOT to use

- **Small samples** — density estimate f̂_Y(q_τ) noisy in tails.
- **Discrete or heaped Y** — density undefined; use smoothed
  distribution regression instead.
- **Target is conditional quantile** — use standard QR
  (Koenker-Bassett).

## Assumptions & caveats

- **Density at q_τ** must be nonzero and consistently estimated;
  bandwidth choice matters, especially at extreme τ.
- **Linear approximation** to the (nonlinear) RIF-conditional-mean;
  extensions include RIF-logit or nonparametric RIF regressions.
- **Interpretation** — RIF-OLS slopes are LOCAL effects at the
  current marginal quantile; large policy shifts require
  counterfactual reweighing (DFL 1996).
- **Standard errors** — bootstrap; analytical SE need influence-
  function corrections for two-step nature.

## Related in this repo

- `quantile-regression`, `bayesian-quantile-regression`,
  `additive-quantile-regression`, `censored-quantile-regression` —
  conditional quantile cousins.
- `quantile-treatment-effects`, `oaxaca-blinder` — decomposition
  toolkit.
- `expectile-regression`, `distributional-regression`, `gamlss` —
  alternative distributional models.
- `influence-functions-eif` — the RIF is a special influence function.

## Run

```
python techniques/rif-regression-firpo/python/rif_regression_firpo.py
Rscript techniques/rif-regression-firpo/r/rif_regression_firpo.R
```

**Refs:** Firpo, S., Fortin, N.M. & Lemieux, T. "Unconditional quantile regressions." *Econometrica* 77(3): 953-973, 2009; Firpo, S., Fortin, N.M. & Lemieux, T. "Decomposing wage distributions using recentered influence function regressions." *Econometrics* 6(2): 28, 2018.

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
