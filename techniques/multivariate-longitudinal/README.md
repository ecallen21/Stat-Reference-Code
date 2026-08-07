# Multivariate Longitudinal Analysis (Reference §12.14)

Two or more longitudinal outcomes measured on the same subjects over time. Analyzing them separately loses cross-outcome dependence and gives inefficient inference. Joint modelling captures:

- correlated **random effects** across outcomes,
- correlated **residuals** within a visit,
- potentially **different fixed-effect structures** per outcome.

## Bivariate LMM formulation

```
y_1,ij = X_1,ij β_1 + Z_1,ij b_1,i + e_1,ij
y_2,ij = X_2,ij β_2 + Z_2,ij b_2,i + e_2,ij

(b_1,i, b_2,i) ~ N(0, D)          joint random-effect covariance
(e_1,ij, e_2,ij) ~ N(0, Σ)         within-visit residual covariance
```

The between-outcome random-effect correlation is the most substantively interesting quantity — it measures the extent to which subjects who are systematically high on outcome 1 are also systematically high on outcome 2, over and above shared covariates.

## Two-stage vs full joint MLE

- **Full joint MLE** — fit both outcomes together via a stacked long-form regression with an outcome-indexed random effect. Correct standard errors, correct joint likelihood; use `nlme::lme` or `brms::brm(mvbind(y1, y2) ~ ...)`.
- **Two-stage approximation** — fit each outcome separately, then correlate the empirical BLUPs. Cheap; biases the correlation slightly toward zero (shrinkage attenuation). Good starting point / sanity check.

## Files

- `python/multivariate_longitudinal.py` — two-stage random-intercept bivariate estimator using shrinkage BLUPs. Demo (N = 100 subjects, T = 5 visits each): recovers cross-outcome random-intercept correlation 0.63 (true 0.60), both slopes within 0.02 of truth.
- `r/multivariate_longitudinal.R` — stacked bivariate `nlme::lme` fit. Production: `brms::brm(mvbind(y1, y2) ~ ...)` for the full joint Bayesian model.

## When to use

- Two clinical outcomes measured jointly (e.g. depression and anxiety scores) where the correlation between the person-level effects is the parameter of interest.
- Bivariate growth-curve analysis with joint inference on the two growth trajectories.
- Any longitudinal study where power for a downstream cross-outcome test benefits from full-information joint modelling.

## Related methods

- **Joint longitudinal-survival models** (§12.10, deferred): tie a longitudinal biomarker to a survival endpoint.
- **Multivariate GEE** (`geepack` / `geeM`): population-average version without random effects.
- **Latent-variable / SEM approaches** (Ch 19): put a latent construct behind the observed outcomes.

## Assumptions & caveats

- Normality of the joint random effects; use `lme4::lmer` with `control = lmerControl(check.nobs.vs.rankZ = "ignore")` or a Bayesian fit for skewed / heavy-tailed data.
- The residual covariance matrix `Σ` is often left as compound-symmetric or unstructured; misspecification changes the within-outcome inference more than the cross-outcome correlation.
- Two-stage estimates attenuate the correlation; when close-to-zero correlations matter, use full joint MLE.

## Run

```
python techniques/multivariate-longitudinal/python/multivariate_longitudinal.py
Rscript techniques/multivariate-longitudinal/r/multivariate_longitudinal.R
```

**Refs:** Reinsel, G.C. "Estimation and prediction in a multivariate random effects generalized linear model." *JASA* 79(386), 406–414, 1984; Fieuws, S. & Verbeke, G. "Pairwise fitting of mixed models for the joint modeling of multivariate longitudinal profiles." *Biometrics* 62(2), 424–431, 2006.

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
