# Instrumental Variables / 2SLS (Reference §5.22)

**Endogeneity** — a covariate `X_endog` correlated with the error term — biases OLS. Common causes:

- **Omitted variables**: a confounder in the error term.
- **Reverse causation**: `y → X_endog`.
- **Measurement error** in `X_endog`.

An **instrument** `Z` fixes this if it satisfies:

- **Relevance**: `Z` is correlated with `X_endog` (conditional on exogenous covariates).
- **Exclusion**: `Z` affects `y` only through `X_endog` (given the model).

## Two-stage least squares (2SLS)

```
Stage 1:  X̂ = OLS(X_endog on Z + X_exog)     — project out the exogenous variation
Stage 2:  β̂_IV = OLS(y on X̂ + X_exog)         — regress on the projected values
```

Consistent for the causal effect of `X_endog` under the two assumptions above.

## Standard errors

Naive Stage-2 SEs are wrong (residuals use `X̂`, not `X`). Correct:

```
Cov(β̂_IV) = σ̂² · (X̂ᵀX̂)⁻¹        σ̂² from residuals  y − X_orig β̂_IV
```

## Weak-instrument diagnostic

- **First-stage F** on the excluded instruments; rule of thumb `F ≥ 10` (Staiger-Stock 1997) for a single endogenous regressor.
- **Weak instruments** — bias approaches OLS bias, SE explodes, coverage is wrong.
- **Anderson-Rubin CI** is robust to weak instruments and should be reported alongside the Wald CI when `F` is borderline.

## Files

- `python/iv_2sls.py` — from-scratch 2SLS with correct SE and first-stage F-diagnostic. Demo (n = 500, true effect 2.0, unobserved confounder): OLS gives 2.14 (biased); 2SLS with instrument `z` gives 1.996; first-stage F = 1473.
- `r/iv_2sls.R` — `AER::ivreg` (also reports Sargan overidentification test, Wu-Hausman endogeneity test, and weak-instrument diagnostics).

## When to use

- **Randomized encouragement designs** — `Z` = random encouragement to take treatment.
- **Natural experiments** — `Z` = lottery, policy discontinuity, quarter-of-birth.
- **Judge-leniency / examiner IV** — random assignment of judges with different leniency rates instruments for outcomes downstream of the judge decision.
- **Mendelian randomization** — genetic variants instrument for exposures whose observational effect is confounded.

## Diagnostics to report

- **First-stage F** — for each endogenous regressor.
- **Sargan / Hansen J-test** — when you have more instruments than endogenous regressors, tests overidentification (i.e. exclusion).
- **Wu-Hausman** — tests whether OLS = IV (i.e. whether endogeneity matters at all).
- **Reduced-form** — regression of `y` on `Z` directly; the sign/significance should match `β̂_IV`.

## Assumptions & caveats

- **Exclusion cannot be tested from data** (with just-identified models); it's a substantive assumption. Justify it explicitly.
- **LATE interpretation**: with heterogeneous effects, IV estimates the **local** average treatment effect — average effect among **compliers** (subjects whose treatment status changes with `Z`).
- **Bias of 2SLS toward OLS** grows with the number of instruments; use LIML or JIVE with many weak instruments.

## Run

```
python techniques/iv-2sls/python/iv_2sls.py
Rscript techniques/iv-2sls/r/iv_2sls.R
```

**Refs:** Wright, P.G. *The Tariff on Animal and Vegetable Oils*, Appendix B, Macmillan, 1928 (first published IV); Angrist, J.D. & Imbens, G.W. "Identification and estimation of local average treatment effects." *Econometrica* 62(2), 467–475, 1994; Staiger, D. & Stock, J.H. "Instrumental variables regression with weak instruments." *Econometrica* 65(3), 557–586, 1997.

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
