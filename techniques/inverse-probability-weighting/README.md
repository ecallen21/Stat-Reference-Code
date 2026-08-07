# Inverse Probability of Treatment Weighting (Reference §15.7)

Alternative to matching for causal inference under strong ignorability. **Reweight** the observed sample to create a pseudo-population where treatment is independent of `X`.

## Weights

```
ATT:  w_i = T_i + (1 − T_i) · e(X_i) / (1 − e(X_i))
ATE:  w_i = T_i / e(X_i) + (1 − T_i) / (1 − e(X_i))
```

where `e(x) = Pr(T = 1 | X = x)` is the propensity score. **Stabilized** weights multiply by the marginal `Pr(T = t)` — reduces variance at the cost of a small change in the target estimand.

## Two IPTW estimators

- **Horvitz-Thompson**: `mean(w_i T_i Y_i − w_i (1 − T_i) Y_i)`. Unbiased but noisy with extreme weights.
- **Hájek** (self-normalized): ratio of weighted means. More stable when weights vary.

## AIPW / Doubly-Robust (Robins-Rotnitzky-Zhao 1994)

Combine outcome regression `μ̂_1, μ̂_0` with IPW:

```
ATE_DR = mean( μ̂_1(X) − μ̂_0(X)
             + T (Y − μ̂_1(X)) / e(X)
             − (1 − T) (Y − μ̂_0(X)) / (1 − e(X)) )
```

**Double robustness**: consistent if **either** the outcome model or the propensity model is correctly specified.

## Files

- `python/inverse_probability_weighting.py` — logistic propensity + ATT/ATE weights (stabilized option) + Horvitz-Thompson and Hájek estimators + AIPW/DR. Demo (n = 800, true ATE 2.0): naive 2.29; IPTW-Hájek 2.11; IPTW-ATT-Hájek 2.05; AIPW 2.10.
- `r/inverse_probability_weighting.R` — `WeightIt::weightit(method = "ps", estimand = "ATE")` + `survey::svyglm` for weighted regression.

## When to use

- Same setting as PSM but you don't want to subset: **keep all controls with a weight**.
- Weighted regression / weighted logistic downstream — full-data efficiency.
- Time-varying treatment (marginal structural models) — IPW generalizes to sequential treatments.
- **DR** as the default when outcome-model + propensity-model are both plausible.

## When NOT to use

- **Extreme weights** — some treated units have propensity near 0 (or controls near 1); Hájek helps but doesn't fix violations of common support. Truncate/trim as a sensitivity check.
- **Small samples** — IPW's asymptotic properties kick in slowly.
- **Unmeasured confounding** — IPW can't fix what isn't in `X`.

## Assumptions & caveats

- Same as PSM: **strong ignorability + SUTVA + common support**.
- **Report weight distribution** (max, mean, ESS = `(Σw)² / Σw²`); ESS below 30% of `n` is a red flag.
- **Sensitivity analysis** — E-value or Rosenbaum bounds for unmeasured confounding.

## Run

```
python techniques/inverse-probability-weighting/python/inverse_probability_weighting.py
Rscript techniques/inverse-probability-weighting/r/inverse_probability_weighting.R
```

**Refs:** Rosenbaum, P.R. "Model-based direct adjustment." *JASA* 82(398), 387–394, 1987; Robins, J.M., Rotnitzky, A. & Zhao, L.P. "Estimation of regression coefficients when some regressors are not always observed." *JASA* 89(427), 846–866, 1994; Hernán, M.A. & Robins, J.M. *Causal Inference: What If*, CRC, 2020.

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
