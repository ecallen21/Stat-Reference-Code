# Regression Discontinuity Design (Reference §15.9)

Treatment assigned by a **known cutoff** on a continuous running variable `R`:

```
Sharp:  T = 1 if R ≥ c, else 0                        deterministic
Fuzzy:  Pr(T = 1) jumps at R = c but not to 1        probabilistic
```

Under continuity of `E[Y(0) | R]` and `E[Y(1) | R]` at `c`, the local treatment effect at the cutoff is identified by the **discontinuity**:

```
Sharp:  τ_SRD = lim_{r→c+} E[Y | R=r] − lim_{r→c−} E[Y | R=r]
Fuzzy:  τ_FRD = (Y jump at c) / (T jump at c)         Wald / IV form
```

## Local-linear estimator (Hahn-Todd-Van der Klaauw 2001)

Fit two **local linear** regressions on either side of the cutoff, using **kernel-weighted** observations within a bandwidth `h`. Report the intercepts and their difference.

- **Triangular kernel** for optimal MSE.
- **Bandwidth**: Imbens-Kalyanaraman plug-in or CV.
- Report several bandwidths for sensitivity.

## Files

- `python/regression_discontinuity.py` — local-linear sharp and fuzzy RDD with a triangular kernel and Silverman rule-of-thumb bandwidth. Demo (n = 500, true τ = 2): sharp τ̂ = 2.05 – 2.25 across bandwidths 0.4 – 1.5; fuzzy τ̂_FRD = 2.11 with 75% compliance.
- `r/regression_discontinuity.R` — `rdrobust::rdrobust` (Calonico-Cattaneo-Titiunik canonical implementation with robust bias-corrected CIs).

## When to use

- **Age / test-score / eligibility cutoffs** — enrollment in a program, scholarship, tax bracket, retirement threshold.
- **Geographic borders** — administrative boundary as the running variable.
- **Time-of-adoption thresholds** — subjects born on / after a policy date.

## Sharp vs fuzzy

- **Sharp**: passing the cutoff **deterministically** decides treatment (e.g. legal age).
- **Fuzzy**: crossing the cutoff **increases the probability** of treatment (compliance is partial). Estimated as a Wald ratio.

## Assumptions & caveats

- **No manipulation of `R`** at the cutoff — test with the McCrary (2008) density discontinuity test.
- **Continuous potential outcomes at `c`** — the untestable RDD identification assumption.
- **Local nature**: RDD identifies the effect **only at the cutoff**. Extrapolating away is heroic.
- **Bandwidth sensitivity**: report multiple bandwidths; use bias-corrected robust CIs (Calonico-Cattaneo-Titiunik 2014).

## Related methods

- **Difference-in-Differences** (`diff-in-diff`) — different identification strategy (parallel trends over time).
- **Instrumental variables** (`iv-2sls`) — fuzzy RDD is a local IV.
- **Synthetic control** (`synthetic-control`) — different identification, single treated unit over time.

## Run

```
python techniques/regression-discontinuity/python/regression_discontinuity.py
Rscript techniques/regression-discontinuity/r/regression_discontinuity.R
```

**Refs:** Hahn, J., Todd, P. & Van der Klaauw, W. "Identification and estimation of treatment effects with a regression-discontinuity design." *Econometrica* 69(1), 201–209, 2001; Imbens, G. & Kalyanaraman, K. "Optimal bandwidth choice for the regression discontinuity estimator." *Rev. Econ. Stud.* 79(3), 933–959, 2012; Calonico, S., Cattaneo, M.D. & Titiunik, R. "Robust nonparametric confidence intervals for regression-discontinuity designs." *Econometrica* 82(6), 2295–2326, 2014.

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
