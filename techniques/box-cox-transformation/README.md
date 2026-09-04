# Box-Cox Transformation (Reference §41.1)

Box & Cox (1964). Power transformation that approximately normalises
and stabilises the variance of a positive response:

```
y(λ) = (y^λ − 1) / λ         if λ ≠ 0
     = log(y)                if λ = 0
```

Special cases: `λ = −1` reciprocal, `0` log, `0.5` square root,
`1` no transformation, `2` square. Choose `λ` by profile log-
likelihood (MLE under normality of transformed residuals).

## When to use

- **Positive response** with skew or heteroscedasticity you want to
  fix before OLS / normal-model inference.

## When NOT to use

- **Zeros or negatives** — Box-Cox is undefined; use Yeo-Johnson
  (see `yeo-johnson-transformation`).
- **When log is theoretically motivated** — take log directly and
  don't waste a df on picking `λ`.
- **Interpretability required** on the original scale — the
  transformed model reports on `y(λ)`, not `y`.

## Files

- `python/box_cox_transformation.py` — profile-likelihood scan of λ
  over [−2, 2] + Shapiro normality check. Demo (n=400, log-normal
  y): **MLE λ̂ = 0.02** (correctly identifies log); Shapiro p
  moves from **3 × 10⁻²⁰ → 0.35**; skewness 2.15 → 0.005.
- `r/box_cox_transformation.R` — `MASS::boxcox`,
  `car::powerTransform`, `bestNormalize` (R); `scipy.stats.boxcox`,
  `sklearn.PowerTransformer` (Python).

## Assumptions & caveats

- **Positive support** required.
- **Normal-error assumption** underlies the profile likelihood; the
  MLE `λ` is optimal under that model.
- **Report the transformation explicitly** and back-transform
  predictions if the audience needs original units.
- **Different `λ` per subgroup** may indicate model
  misspecification; check residuals per group.

## Related in this repo

- `yeo-johnson-transformation` — extension to zero and negative
  values.
- `inverse-normal-transformation` — rank-based force to exact
  normality.
- `standardization-scaling` — post-transform standardisation.

## Run

```
python techniques/box-cox-transformation/python/box_cox_transformation.py
Rscript techniques/box-cox-transformation/r/box_cox_transformation.R
```

**Refs:** Box, G.E.P. & Cox, D.R. "An analysis of transformations." *JRSS-B*, 1964; Sakia, R.M. "The Box-Cox transformation technique: a review." *The Statistician*, 1992.

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
