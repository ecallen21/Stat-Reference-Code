# Pattern-Mixture Models for MNAR (Reference §16.x extra)

Missing-at-random (MAR) is the workhorse assumption behind `mice`,
`multiple-imputation`, and full-information ML. **Pattern-mixture models**
handle **missing-not-at-random (MNAR)** by factoring the joint distribution
by the missingness pattern `R`:

```
f(Y, R) = f(Y | R) · f(R)
```

`f(Y | R = 1)` — the distribution of the missing responses — is
**unidentifiable** from observed data alone. Pattern-mixture requires a
**sensitivity parameter** `δ` to bridge:

```
E[Y | R = 1] = E[Y | R = 0] + δ
```

- `δ = 0` → MAR (missingness is "just like" the observed with the same predictors).
- `δ > 0` → missing tend to be higher than observed with the same predictors (MNAR-up).
- `δ < 0` → missing tend to be lower (MNAR-down).

The **delta-adjustment MI** procedure imputes under MAR, then shifts each
imputed value by `δ`. Report estimates across a plausible range of `δ` —
**tipping-point analysis** identifies the smallest `|δ|` that flips a
conclusion.

## Related MNAR strategies

| Approach | Idea |
|---|---|
| **Selection model** (Diggle-Kenward, Heckman) | model `f(R | Y)` explicitly; jointly ML |
| **Pattern-mixture** (this module) | stratify on `R`, sensitivity parameter |
| **Shared parameter** | latent variable drives both `Y` and dropout |
| **Reference-based imputation** (Carpenter-Kenward 2013) | impute dropouts from a reference group (e.g. control-arm mean) |
| **Multiple imputation with delta** | practical tipping-point sensitivity used in FDA/EMA submissions |

## When to use

- **Clinical trials with informative dropout** — regulator-recommended sensitivity analysis.
- **Longitudinal surveys** where non-response is likely related to the outcome.
- **Any observational study** where you suspect missingness depends on the outcome itself.
- **Sensitivity check** for a headline MAR-based analysis.

## Files

- `python/pattern_mixture_model.py` — delta-adjusted MI: MAR imputation via OLS + Gaussian residual, then shift imputed `y` by `δ`. Demo (n=400, x continuous, MNAR mechanism: higher `y` → more likely missing; true `E[Y]` = 1.93): complete-case mean 1.25 (badly biased low), MAR imputation (`δ=0`) 1.67 (still biased), `δ=+1` gives 2.02 recovering the truth — the tipping delta.
- `r/pattern_mixture_model.R` — `mice` with `post` argument for delta, `jomo::jomo1ranmix`, `SensMice`, `RefBasedMI`.

## Assumptions & caveats

- **`δ` is fundamentally unverifiable** from the data — its range must come from subject-matter reasoning (previous studies, expert elicitation).
- **Report the full sensitivity range**, not a single δ. The tipping point (where the conclusion flips) is often more informative than a point estimate.
- **Model dependence** — MAR imputation model still matters; a mis-specified conditional mean carries into pattern-mixture too.
- **Multivariate outcomes** need careful joint imputation (`jomo`, `mice` with pmm); univariate delta shifts can distort correlations.
- **Reference-based** methods (Carpenter-Kenward) are a common alternative in clinical trials, and are closer in spirit to pattern-mixture than to plain MI.

## Related in this repo

- `multiple-imputation` — MI under MAR.
- `heckman-selection` — parametric selection model (MNAR alternative).
- `sensitivity-e-value` — quantify unmeasured-confounding-style robustness for a causal estimand.

## Run

```
python techniques/pattern-mixture-model/python/pattern_mixture_model.py
Rscript techniques/pattern-mixture-model/r/pattern_mixture_model.R
```

**Refs:** Little, R.J.A. "Pattern-mixture models for multivariate incomplete data." *JASA* 88(421), 125–134, 1993; Carpenter, J.R. & Kenward, M.G. *Multiple Imputation and its Application*, Wiley, 2013; Cro, S. et al. "Sensitivity analysis for clinical trials with missing continuous outcome data using controlled multiple imputation: a practical guide." *Stat. Med.* 39(21), 2815–2842, 2020.

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
