# Measurement Invariance (Reference §19.x extra)

Tests whether a factor model has the **same meaning** across groups (culture,
language, gender, time). Requirement for legitimate group-mean comparisons —
otherwise a difference in observed scores could reflect a difference in the
instrument rather than the construct.

## The invariance chain

| Level | Constrained equal across groups | Enables |
|---|---|---|
| **Configural** | same factor structure only | qualitative comparison |
| **Metric** (weak) | + loadings `λ` | "same meaning" of factor scores |
| **Scalar** (strong) | + intercepts `τ` | latent-mean comparison |
| **Strict** | + residual variances `θ` | observed-score comparison |

Each level is nested in the previous. Test by chi-square difference between
adjacent levels; modern practice **also** reports:

- `ΔCFI < 0.010` (Cheung & Rensvold 2002),
- `ΔRMSEA < 0.015`,
- practical vs statistical significance (`n` inflates chi²).

If strict fails, look at **partial invariance** (Byrne 1989) — free one
parameter at a time using modification indices.

## When to use

- **Cross-cultural** / cross-language survey research.
- **Longitudinal** — is the measurement the same at each wave?
- **Method effects** — comparing paper vs online administration.
- **Machine-scoring / rater comparisons** — do raters use the scale the same way?
- **Any latent-variable comparison** — invariance is a prerequisite for interpretable group differences.

## Files

- `python/measurement_invariance.py` — from-scratch ML fits of the two extreme models (configural + strict) for a single-factor CFA; nested LR test. Demo (p=4 indicators, two groups n=400 each):
  - **Invariant DGP** (group 2 differs only in factor mean κ=0.5): configural −2LL = 2067.9, strict = 2071.9, χ²_diff = 4.0, p = 0.947 — strict invariance is NOT rejected. Strict-model estimates recover true loadings, intercepts, residual vars, and κ ≈ 0.539 vs true 0.5.
  - **Non-invariant DGP** (τ₁ shifted by +1 in group 2): χ²_diff = 414, p < 0.0001 — strict correctly rejected.
- `r/measurement_invariance.R` — `semTools::measurementInvariance`, `semTools::measEq.syntax`, `semTools::partialInvariance`, `lavaan::cfa` + `lavaan::anova`.

## Assumptions & caveats

- **Nested tests are sensitive to `n`** — with large samples even trivial non-invariance rejects; always report `ΔCFI` alongside.
- **Choice of anchor** — the reference indicator (loading fixed to 1, intercept to 0) affects which parameters are free and can drive partial-invariance conclusions.
- **Configural fit** must be adequate to begin with — no point in testing invariance of a badly-fitting model.
- **Categorical indicators** need robust WLSMV estimator and threshold invariance instead of intercept invariance — see `lavaan(... , ordered = ...)`.
- **Longitudinal invariance** requires modelling correlated residuals across waves (autocorrelation) alongside the invariance constraints.
- **Alignment method** (Muthén-Asparouhov 2014) — an alternative when strict invariance is untenable across many groups.

## Related in this repo

- `cfa-confirmatory-factor` — single-group CFA (the building block).
- `path-analysis` — structural relations among latent variables.
- `latent-class-analysis` — invariance for categorical latent variables.
- `bayesian-model-comparison` — an alternative comparison framework.

## Run

```
python techniques/measurement-invariance/python/measurement_invariance.py
Rscript techniques/measurement-invariance/r/measurement_invariance.R
```

**Refs:** Meredith, W. "Measurement invariance, factor analysis and factorial invariance." *Psychometrika* 58(4), 525–543, 1993; Cheung, G.W. & Rensvold, R.B. "Evaluating goodness-of-fit indexes for testing measurement invariance." *Struct. Equ. Modeling* 9(2), 233–255, 2002; Vandenberg, R.J. & Lance, C.E. "A review and synthesis of the measurement invariance literature: suggestions, practices, and recommendations." *Org. Res. Meth.* 3(1), 4–70, 2000.

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
