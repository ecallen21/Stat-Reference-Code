# Quantile IV Regression (Chernozhukov-Hansen) (Reference §15.44)

Chernozhukov & Hansen (2005, 2008). Combines IV identification with
quantile-regression estimation:

    Q_τ(Y | D, X) = D · α(τ) + X · β(τ)

When `D` is endogenous, use instrument `Z`. The CH estimator solves

    α̂(τ) = argmin_a  ‖γ̂(a, τ)‖
    γ̂ = QR-coefficient on Z after subtracting D · a

At the true `α`, `Z` has no residual effect on the conditional
quantile (analogous to IV moment condition).

## Files

- `python/quantile_iv_regression.py` — grid-search CH-QIV for
  scalar `D` from scratch. Demo (n=500, true α=1.0, endogeneity via
  correlated u): naive median regression 0.839 (biased); CH-QIV
  1.000 (correct).
- `r/quantile_iv_regression.R` — `quantreg + ivqte`, `IVQuantile`
  (R); `econml.iv`, from-scratch (Python).

## When to use

- **Endogenous continuous treatment / dose** — heterogeneous
  effects across the outcome distribution.
- **Distributional causal analysis** — QTE at multiple `τ` to
  reveal effect variation.
- **Wage / earnings / bio-response research** — CH is standard.

## When NOT to use

- **Homogeneous effect** — a scalar τ isn't necessary; use 2SLS.
- **Discrete treatment with LATE structure** — use CACE / Firpo
  QTE.
- **Weak instrument** — QIV inherits IV weakness; check first-
  stage F.

## Assumptions & caveats

- **Rank invariance / rank similarity** — CH requires one of these
  to identify structural quantile effects.
- **Valid IV** — same as 2SLS (relevance + exogeneity).
- **Grid vs continuous search** — for scalar α, grid works; for
  higher-dim, use nested optimisation.
- **Standard errors** — bootstrap or the asymptotic sandwich in
  Chernozhukov-Hansen 2006.

## Related in this repo

- `iv-2sls`, `principal-stratification-cace`,
  `weak-instruments-anderson-rubin` — IV cousins.
- `quantile-regression`, `bayesian-quantile-regression`,
  `censored-quantile-regression`, `additive-quantile-regression`
  — quantile family.
- `quantile-treatment-effects` — Firpo IPW alternative.

## Run

```
python techniques/quantile-iv-regression/python/quantile_iv_regression.py
Rscript techniques/quantile-iv-regression/r/quantile_iv_regression.R
```

**Refs:** Chernozhukov, V. & Hansen, C. "An IV model of quantile treatment effects." *Econometrica*, 73(1): 245-261, 2005; Chernozhukov, V. & Hansen, C. "Instrumental variable quantile regression: a robust inference approach." *Journal of Econometrics*, 142(1): 379-398, 2008.

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
