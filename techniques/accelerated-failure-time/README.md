# Accelerated Failure Time (AFT) Models (Reference §11.25)

Kalbfleisch & Prentice (2002). Parametric survival regression on the
**log of failure time**:

    log T = X · β + σ · W

with `W` a standard error distribution.

## Common choices

| Error W | Failure time T | Notes |
|---|---|---|
| Extreme value (min) | Weibull | Also PH; special case Exp when σ=1 |
| Logistic | Log-logistic | Non-monotone hazard |
| Normal | Log-normal | Long tails, unimodal hazard |
| Generalised γ | Gen. gamma | Superset of Weibull, log-normal, exp |

## Interpretation

`exp(βⱼ)` is the **acceleration factor** — a unit increase in `Xⱼ`
multiplies survival time by `exp(βⱼ)`. Contrast with Cox, where
`exp(βⱼ)` multiplies the **hazard**.

## Files

- `python/accelerated_failure_time.py` — Weibull and log-normal AFT
  MLE via L-BFGS-B from scratch, with right-censoring. Demo (n=800,
  true β₁=0.5, β₂=−0.8, σ=0.4, 21% censoring): recovered β=+0.512,
  −0.819; σ=0.385; Weibull log-lik = −1274 beats log-normal −1351.
- `r/accelerated_failure_time.R` — `survival::survreg`,
  `flexsurv::flexsurvreg`, `rms::psm` (R);
  `lifelines.WeibullAFTFitter` / `LogNormalAFTFitter` (Python).

## When to use

- **PH assumption fails** — AFT does not require proportional
  hazards; time-varying / crossing hazards fine.
- **Prediction of survival TIME** — AFT gives a direct multiplicative
  interpretation on T.
- **Small-to-moderate n with a good distributional guess** — more
  efficient than Cox when the parametric form is defensible.

## When NOT to use

- **Unknown / misspecified distribution** — the parametric form can
  bias effect estimates; check fits (Q-Q plot on the log scale, AIC
  across distributions, or a generalised-gamma nesting test).
- **Very small samples with tied event times** — the MLE can be
  unstable; consider penalised or Bayesian AFT.
- **Time-varying covariates** — AFT with time-varying X is
  possible but rare; Cox time-varying is more standard.

## Assumptions & caveats

- **Independent censoring** — same as Cox / KM.
- **Correct distribution** — validate with cumulative-hazard plots
  (Weibull: log H vs log T should be linear); consider
  generalised-gamma nesting.
- **Log-linearity** — the log-scale linearity of β·X can be relaxed
  via splines / fractional polynomials in X.
- **Scale parameter σ** — a Weibull with σ=1 is the exponential
  (constant hazard); test the reduction with an LRT.

## Related in this repo

- `parametric-survival` — broader parametric survival techniques.
- `cox-ph`, `cox-time-varying` — semi-parametric alternatives.
- `flexsurv` / `frailty-models` — shared-frailty extensions of AFT.
- `random-survival-forest` — non-parametric alternative for
  prediction.

## Run

```
python techniques/accelerated-failure-time/python/accelerated_failure_time.py
Rscript techniques/accelerated-failure-time/r/accelerated_failure_time.R
```

**Refs:** Kalbfleisch, J.D. & Prentice, R.L. *The Statistical Analysis of Failure Time Data*, 2nd ed., Wiley, 2002; Collett, D. *Modelling Survival Data in Medical Research*, 3rd ed., CRC, 2015.

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
