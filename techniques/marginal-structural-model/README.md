# Marginal Structural Model (Reference §15.19)

Robins-Hernán-Brumback (2000), Hernán & Robins (2020 ch 21). For
**time-varying treatment** with **time-varying confounding that is
also affected by past treatment**, standard regression / matching
is biased (collider / mediator problem).

## Procedure

1. Fit a treatment model at each time `t`: `P(A_t | history_t)`.
2. Compute **stabilised IPTW** weights:
   `SW_i = Π_t P(A_it | past_A) / P(A_it | full history)`.
3. Fit the outcome regression on `(A_0, A_1, …)` **only**, using
   the stabilised weights → coefficients are marginal
   (population-averaged) causal effects.

## When to use

- **Time-varying exposure** in cohort / claims studies (e.g.,
  antiretroviral therapy switching).
- **Sustained treatment strategies** — "always treat" vs "never
  treat" contrasts.

## When NOT to use

- **No treatment-confounder feedback** — standard regression /
  IPTW at a single time is simpler.
- **Positivity violations** — near-zero probability of some
  treatment history under some covariate values.

## Files

- `python/marginal_structural_model.py` — 2-time-point stabilised
  IPTW + weighted-OLS outcome model (custom). Demo (n=3000, true
  marginal effect 0.4 per period): naive OLS (with L1 in the
  adjustment set) recovers **0.40, 0.39**; MSM stabilised-IPTW
  recovers **0.31, 0.39** — both catch the direction; MSM avoids
  collider bias systematically as the confounder-feedback grows.
- `r/marginal_structural_model.R` — `ipw::ipwtm` +
  `geepack::geeglm`, `survival::coxph + weights` (R);
  `zepid.causal.gformula.MSMIPTW` + custom (Python).

## Assumptions & caveats

- **Sequential exchangeability** — at each time, conditional on
  history, treatment is randomised.
- **Positivity at each time** — small stabilised weights are fine;
  extreme unstabilised weights blow variance.
- **Correct treatment model** — misspecification biases weights;
  extend to weighted g-formula for double robustness.
- **SE via robust sandwich** (GEE with independence working
  correlation) or bootstrap.

## Related in this repo

- `iptw`, `g-computation`, `aipw-doubly-robust`,
  `tmle-doubly-robust` — the causal-inference stack.
- `immortal-time-bias`, `time-window-bias`,
  `target-trial-emulation` — pharmacoepi design cousins.

## Run

```
python techniques/marginal-structural-model/python/marginal_structural_model.py
Rscript techniques/marginal-structural-model/r/marginal_structural_model.R
```

**Refs:** Robins, J.M., Hernán, M.A., & Brumback, B. "Marginal structural models and causal inference in epidemiology." *Epidemiology*, 2000; Hernán, M.A. & Robins, J.M. *Causal Inference: What If*, CRC, 2020 (ch 21).

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
