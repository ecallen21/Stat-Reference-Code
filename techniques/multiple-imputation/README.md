# Multiple Imputation by Chained Equations (Reference §18.6)

Missing-data problem: complete-case analysis loses power; **single** imputation understates uncertainty about the missing values. **Multiple imputation** (Rubin 1987) fixes both.

## Recipe

1. Generate `M` complete datasets by drawing missing values from their **predictive posterior** given observed data.
2. Run the analysis on each dataset → `M` estimates `(θ̂_m, U_m)`.
3. Combine via **Rubin's rules**:

```
θ̂_pool = mean(θ̂_m)
W̄      = mean(U_m)                        within-imp variance
B       = var(θ̂_m)                        between-imp variance
T       = W̄ + (1 + 1/M) B                 total variance
df      ≈ Barnard-Rubin adjusted df
```

## MICE — chained equations (Van Buuren 2007)

For each variable with missing values, model it as a function of the other variables (Bayesian regression, predictive mean matching, logistic for binary, ...), cycle through until stability. Produces one imputed dataset per cycle. Robust default in `mice` (R) and `IterativeImputer` (sklearn).

## Files

- `python/multiple_imputation.py` — from-scratch MICE with Bayesian linear regression per column + Rubin combining rules. Demo (n = 300, 30% MAR missing in `x2`): complete-case β = (1.07, 2.02, -0.91); MICE + Rubin pool β = (1.04, 2.02, -0.87) close to true (1, 2, -1); reports fraction-of-missing-info (FMI) per coefficient.
- `r/multiple_imputation.R` — `mice::mice(data, m = 10)` + `mice::pool` (the canonical R implementation).

## When to use

- Any analysis with **missing at random (MAR)** covariates or outcome — the standard case for most observational studies.
- Regression, GLM, survival, mixed models, PSM — MI applies to any downstream analysis.

## When NOT to use

- **MNAR (missing not at random)** — the missingness mechanism depends on the unobserved value itself. MI is biased; use pattern-mixture / selection models with substantive assumptions.
- **Very large `p` with small `n`** — chained equations don't fit; consider joint MI (`Amelia`) or a Bayesian latent-model imputer.

## Rubin's assumptions

- **MAR** — missingness depends only on observed data.
- **Congeniality** — imputation model and analysis model are compatible; include all analysis-model variables in the imputer.
- **Proper imputation** — draws from the posterior predictive, not from the ML estimate (that would understate uncertainty).

## Diagnostics

- **Fraction of missing information (FMI)**: `λ ≈ (1 + 1/M) B / T`. Rough interpretation: with FMI = 30%, running with `M = 20` costs the same as `M = 10` in a lower-FMI setting.
- **Convergence traces** of the chained equations for each variable — should stabilize.
- **Density plots** of observed vs imputed values per variable — should overlap in the MAR case.

## Run

```
python techniques/multiple-imputation/python/multiple_imputation.py
Rscript techniques/multiple-imputation/r/multiple_imputation.R
```

**Refs:** Rubin, D.B. *Multiple Imputation for Nonresponse in Surveys*, Wiley, 1987; Van Buuren, S. & Groothuis-Oudshoorn, K. "MICE: multivariate imputation by chained equations in R." *J. Stat. Softw.* 45(3), 1–67, 2011; Van Buuren, S. *Flexible Imputation of Missing Data*, 2nd ed., CRC, 2018.

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
