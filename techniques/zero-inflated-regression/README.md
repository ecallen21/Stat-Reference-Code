# Zero-Inflated and Hurdle Count Regression (Reference §5.24)

Count outcomes with **more zeros than Poisson / Negative-Binomial predicts** — dental caries counts, insurance claims, species abundance, healthcare utilization. Two model families.

## Zero-inflated Poisson (Lambert 1992)

```
y_i = 0                with probability π_i        (structural zero)
y_i ~ Poisson(μ_i)     with probability (1 − π_i)  (potentially zero)

logit π_i = Z_i γ        (zero-inflation submodel)
log   μ_i = X_i β        (count submodel)
```

The count submodel can itself produce zeros, so ZIP has **two paths to zero**: structural (`π_i`) plus sampling.

## Hurdle Poisson (Cragg 1971)

```
y_i = 0             with probability π_i
y_i | y_i > 0 ~ TruncPoisson(μ_i)      (zero-truncated)
```

Exactly one path to zero. Cleanly separates "did any event occur?" from "how many, given at least one?". Fits as two independent GLMs.

## ZIP vs hurdle

- **ZIP**: interpret as a latent binary "susceptibility" indicator; use when some subjects are structurally immune (never-hunters, non-drinkers).
- **Hurdle**: use when the zero-vs-positive decision is a distinct process from the count intensity (visit-vs-not-visit, then intensity of visits).

Swap Poisson → Negative-Binomial for over-dispersion → **ZINB** / **hurdle-NB**.

## Files

- `python/zero_inflated_regression.py` — from-scratch BFGS MLE for both ZIP and hurdle Poisson. Demo (n = 500, 30% structural zeros): ZIP recovers all four parameters matching `statsmodels.discrete.count_model.ZeroInflatedPoisson` to 3 decimals.
- `r/zero_inflated_regression.R` — `pscl::zeroinfl` and `pscl::hurdle`.

## When to use

- **Excess zeros**: histogram spike at 0 relative to a Poisson / NB fit; Vuong test rejects Poisson in favor of ZIP.
- **Over-dispersion after ZIP**: switch to ZINB.
- **Interpretable zero process**: the zero-inflation covariates `Z` describe susceptibility separately from intensity.

## Assumptions & caveats

- Identifiability: `Z` and `X` can overlap but shouldn't be identical unless well-anchored — otherwise `π` and `μ` trade off.
- Model selection: compare ZIP vs hurdle vs Poisson vs NB with AIC / BIC / Vuong test.
- Bayesian variants: `brms::brm(y ~ x, family = zero_inflated_poisson())`.

## Run

```
python techniques/zero-inflated-regression/python/zero_inflated_regression.py
Rscript techniques/zero-inflated-regression/r/zero_inflated_regression.R
```

**Refs:** Lambert, D. "Zero-inflated Poisson regression, with an application to defects in manufacturing." *Technometrics* 34(1), 1–14, 1992; Cragg, J.G. "Some statistical models for limited dependent variables with application to the demand for durable goods." *Econometrica* 39(5), 829–844, 1971; Vuong, Q.H. "Likelihood ratio tests for model selection and non-nested hypotheses." *Econometrica* 57(2), 307–333, 1989.

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
