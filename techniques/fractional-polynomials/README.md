# Fractional Polynomials (Reference §5.14)

Royston & Altman (1994). Flexible **parametric** alternative to
splines for modelling nonlinear covariate effects, restricted to the
fixed power set `P = {−2, −1, −0.5, 0, 0.5, 1, 2, 3}` (0 = log).

## Forms

    FP1:  f(x) = β · x^p
    FP2:  f(x) = β₁ · x^{p₁} + β₂ · x^{p₂}          (p₁ ≠ p₂)
    FP2:  f(x) = β₁ · x^p    + β₂ · x^p · log(x)    (p₁ = p₂)

Best-fitting FP1 is chosen from **8** models, FP2 from **36**. Model
choice via the **Royston-Sauerbrei function-selection procedure** —
closed test comparing FP2 vs FP1 vs linear vs null at level α.

## Files

- `python/fractional_polynomials.py` — brute-force power search +
  RS closed test from scratch. Demo (n=400, true `y = 3 − 4/√x +
  0.02·x`): FP1 picks p=0; FP2 picks (−0.5, +0.5), recovering the
  `1/√x` term exactly. Closed test selects FP2.
- `r/fractional_polynomials.R` — `mfp::mfp`, `mfp::mfp2`, `mfpa` (R);
  from-scratch (Python; no mature FP library).

## When to use

- **Parsimonious nonlinear fits** — 1–2 terms with interpretable
  power exponents are easier to report than a spline basis.
- **Clinical prediction models** — TRIPOD-recommended and widely
  used in prognosis / risk-score literature (Royston-Sauerbrei
  2008).
- **Small n** — FP has fewer degrees of freedom than a spline of
  comparable flexibility.

## When NOT to use

- **Very wiggly curves** — the 8-power set is limited; a natural
  cubic spline captures complex shapes better.
- **Non-positive x** — powers like log or negative exponents require
  `x > 0`; add a shift or use splines.
- **You want a smooth-CI band** — FP fit is discrete; bootstrap the
  power selection for honest inference.

## Assumptions & caveats

- **x > 0** — shift `x → x − x_min + δ` before applying FP with
  non-integer / log powers.
- **Selection variance** — the chosen `(p₁, p₂)` are themselves data
  driven; naive CIs ignore this. Bootstrap the full FP procedure.
- **Alpha level** — the closed test uses a nominal α; the RS-2008
  recommendation is α = 0.05 for the FP2-vs-FP1 step but stricter
  overall.
- **Interpretation** — coefficients on `x^p` are on the transformed
  scale; report plotted `f̂(x)` with a CI, not the coefficient.

## Related in this repo

- `splines-regression`, `additive-quantile-regression`, `gam` —
  non-parametric alternatives.
- `polynomial-regression` — the classical (integer-power) special case.
- `multivariable-model-building` — the wider TRIPOD-style clinical
  prediction pipeline.

## Run

```
python techniques/fractional-polynomials/python/fractional_polynomials.py
Rscript techniques/fractional-polynomials/r/fractional_polynomials.R
```

**Refs:** Royston, P. & Altman, D.G. "Regression using fractional polynomials of continuous covariates: parsimonious parametric modelling." *JRSS-C*, 43(3): 429-467, 1994; Royston, P. & Sauerbrei, W. *Multivariable Model-building with Fractional Polynomials*, Wiley, 2008.

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
