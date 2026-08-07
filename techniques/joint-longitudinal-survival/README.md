# Joint Longitudinal-Survival Model (Reference §12.10)

Longitudinal biomarker `y_ij` measured over time on subject `i`, plus a survival outcome `(T_i, δ_i)`. The biomarker is measured with error, is often observed less frequently near the event, and is related to the hazard of the event. All three features bias standard analyses.

## Why not the naive alternatives?

- **Cox with last-observation-carried-forward** — assumes visit spacing is non-informative; typically not true near events.
- **Cox with time-varying observed `y_ij`** — no measurement-error correction; attenuates the association toward the null.
- **Two-stage without full likelihood** — fit LME first, plug BLUPs into Cox — biased when the true association `α` is strong (informative-censoring feedback).

## Joint model (Tsiatis-Davidian 2004)

```
y_ij = X_ij β + m_i(t_ij) + ε_ij           m_i(t) = z_i(t) b_i    (random effect)
h_i(t) = h_0(t) · exp(α · m_i(t) + γᵀ W_i)
```

`m_i(t)` is the true, latent biomarker trajectory. `α` is the association parameter of clinical interest: **how much does a one-unit change in the current biomarker level multiply the hazard?**

## Estimation

- **Full joint MLE** (`JM::jointModel`) — integrate over the random effects via Gauss-Hermite quadrature. Correct, expensive.
- **Full joint Bayesian** (`JMbayes2::jm`) — MCMC; handles multiple biomarkers and non-Gaussian error naturally.
- **Two-stage approximation** — fit LME first, plug BLUPs into Cox with time-varying `m̂_i(t)`. Biased for strong `α`; useful as a diagnostic starting point.

## Files

- `python/joint_longitudinal_survival.py` — two-stage joint model with random-intercept + slope LME and Cox partial likelihood with time-varying `m̂_i(t)`. Demo (N = 150, α_true = 0.8): two-stage recovers `α̂ = 0.50` (attenuated by the two-stage bias), matching the naive `y(0)` fit at `0.51` — full joint MLE is needed for unbiased `α`.
- `r/joint_longitudinal_survival.R` — recipe for `JM::jointModel(lme_fit, coxph_fit, timeVar = "time")` and `JMbayes2::jm`.

## When to use

- Any longitudinal biomarker + survival outcome where the biomarker is thought to drive the hazard: HIV viral load, PSA, CD4, HbA1c, tumor size.
- Dynamic prediction: forecast survival given the observed biomarker history up to now.
- Correcting Cox coefficient attenuation caused by measurement error in a time-varying covariate.

## Association structures

`JM` supports several ways `m_i(t)` can enter the hazard:

- **Current value**: `h(t) = h_0(t) exp(α · m_i(t))`.
- **Slope**: hazard depends on `m_i'(t)`.
- **Cumulative effect**: hazard depends on `∫_0^t m_i(u) du`.
- **Random-effect only**: hazard depends on `b_i` (subject-specific intercept / slope).

Choose based on subject-matter rather than fit statistics.

## Assumptions & caveats

- **Random-effect distribution** — Normal is standard; check with QQ plots of the empirical BLUPs.
- **Non-informative visit process** given `m_i(t)` — if visits happen because subjects feel sick, the joint model still needs an explicit visit-process submodel.
- **Computationally heavy** — Gauss-Hermite quadrature costs `O(K^q)` where `q` is the random-effect dimension; keep `q ≤ 3`.
- **Full joint MLE required** for unbiased `α` when the association is strong (`|α| > 0.5` on the log-hazard scale).

## Run

```
python techniques/joint-longitudinal-survival/python/joint_longitudinal_survival.py
Rscript techniques/joint-longitudinal-survival/r/joint_longitudinal_survival.R
```

**Refs:** Wulfsohn, M.S. & Tsiatis, A.A. "A joint model for survival and longitudinal data measured with error." *Biometrics* 53(1), 330–339, 1997; Tsiatis, A.A. & Davidian, M. "Joint modeling of longitudinal and time-to-event data: an overview." *Stat. Sinica* 14(3), 809–834, 2004; Rizopoulos, D. *Joint Models for Longitudinal and Time-to-Event Data*, CRC, 2012.

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
