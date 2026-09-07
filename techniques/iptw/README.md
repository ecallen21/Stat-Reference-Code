# Inverse Probability of Treatment Weighting (Reference §15.6)

Rosenbaum & Rubin (1983), Li-Morgan-Zaslavsky (2018). Given a
propensity score `e(X) = P(T=1|X)`, reweight observations to mimic
a randomised trial.

## Estimand weights

| Estimand | Weight |
|---|---|
| **ATE** (average) | `T/e + (1−T)/(1−e)` |
| **ATT** (on treated) | `T + (1−T) · e/(1−e)` |
| **ATC** (on control) | `T · (1−e)/e + (1−T)` |
| **ATO** (overlap) | `T · (1−e) + (1−T) · e` |

Then the Hájek estimator
`Σ w·T·y / Σ w·T − Σ w·(1−T)·y / Σ w·(1−T)`.

## When to use

- **Observational causal inference** with rich covariate
  adjustment.
- **Comparative effectiveness** in claims / EHR.
- **Overlap weighting (ATO)** — near-violated positivity settings;
  Li et al. show it minimises asymptotic variance.

## When NOT to use

- **Poor overlap** — extreme weights inflate variance; trim or use
  ATO.
- **Time-varying confounding** — extend to Marginal Structural
  Models (see `marginal-structural-model`).

## Files

- `python/iptw.py` — ATE / ATT / ATC / ATO estimators with
  trimming (custom). Demo (n=3000, true ATE=0.5): naive diff
  **0.90 (biased)**; **IPTW-ATE 0.46, ATO 0.47** — all four
  weighting variants recover the truth.
- `r/iptw.R` — `WeightIt::weightit`, `ipw`, `cobalt`,
  `survey::svyglm` (R); `causalinference`, `zepid`, `econml`
  (Python).

## Assumptions & caveats

- **Unconfoundedness** — the PS model must include all
  confounders.
- **Positivity** — every X must have both treated and untreated
  observations; trimming or ATO handles near-violations.
- **PS specification** — misspecification biases IPTW; combine with
  outcome regression → doubly robust (see `aipw-doubly-robust`).
- **SEs need bootstrap or robust sandwich** — naive SEs
  underestimate variance.

## Related in this repo

- `propensity-score-matching` — companion adjustment.
- `aipw-doubly-robust`, `tmle-doubly-robust` — robust extensions.
- `entropy-balancing`, `coarsened-exact-matching` — alternatives
  when PS misspecification is a concern.

## Run

```
python techniques/iptw/python/iptw.py
Rscript techniques/iptw/r/iptw.R
```

**Refs:** Rosenbaum, P.R. & Rubin, D.B. "The central role of the propensity score in observational studies for causal effects." *Biometrika*, 1983; Li, F., Morgan, K.L., & Zaslavsky, A.M. "Balancing covariates via propensity score weighting." *JASA*, 2018.

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
