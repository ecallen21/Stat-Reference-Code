# Repeated-Measures ANOVA + Sphericity Corrections (Reference §12.1)

For `n` subjects each measured under `K` within-subject conditions (or K time points):

```
Total SS  =  SS_subjects  +  SS_condition  +  SS_error
F  =  MS_condition / MS_error   ~  F(K − 1, (n − 1)(K − 1))   under H₀ + sphericity
```

## Sphericity assumption

**Sphericity** = equality of the variances of all pairwise differences between conditions. When it fails, the F test's df are wrong and p-values are miscalibrated.

Two standard corrections to df:

- **Greenhouse–Geisser (GG)** — always applicable; conservative:
  ```
  ε_GG = [tr(A)]² / [(K − 1) · tr(A²)]      A = C·S·C
  ```
- **Huynh–Feldt (HF)** — less conservative when true ε > 0.75:
  ```
  ε_HF = min(1, (n(K − 1)·ε_GG − 2) / ((K − 1)(n − 1 − (K − 1)·ε_GG)))
  ```

**Girden's rule**: if `ε_GG < 0.75` use GG; if `≥ 0.75` use HF.

Multiply BOTH df of F by ε before reading off the p-value.

## Effect size

`partial η² = SS_condition / (SS_condition + SS_error)` — proportion of within-subject variance explained by condition.

## When to prefer LMM over RM-ANOVA

- Missing data (RM-ANOVA drops incomplete subjects; LMM uses all).
- Continuous or time-varying covariates.
- Explicit variance-components interpretation.
- Different subjects on different time schedules.

For all of the above, see [`linear-mixed-models`](../linear-mixed-models).

## Files

- `python/repeated_measures_anova.py` — from-scratch SS decomposition + GG and HF ε + p-values; optional cross-check via `pingouin.rm_anova` when installed.
- `r/repeated_measures_anova.R` — from-scratch + base `stats::aov(y ~ condition + Error(subject/condition))`.

## Assumptions

- **Complete data** per subject (all K conditions). Use LMM for unbalanced designs.
- Independence between subjects; sphericity within subjects (or apply GG/HF).
- Approximately normal residuals within condition.

## Run

```
python techniques/repeated-measures-anova/python/repeated_measures_anova.py
Rscript techniques/repeated-measures-anova/r/repeated_measures_anova.R
```

**Refs:** Girden, E.R. *ANOVA: Repeated Measures*, Sage, 1992; Greenhouse, S.W. & Geisser, S. "On methods in the analysis of profile data." *Psychometrika* 24(2), 95–112, 1959; Huynh, H. & Feldt, L.S. "Estimation of the Box correction for degrees of freedom from sample data in randomized block and split-plot designs." *J. Educ. Stat.* 1(1), 69–82, 1976; Mauchly, J.W. "Significance test for sphericity of a normal n-variate distribution." *Ann. Math. Stat.* 11(2), 204–209, 1940.

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
