# Rao-Blackwellization (Reference §45.11)

Rao (1945); Blackwell (1947). Given an estimator `T(X)` and a
sufficient statistic `S(X)`, the conditional expectation

    T*(S) = E[T(X) | S(X)]

is at least as good in mean-square error:

    Var(T*(S))  ≤  Var(T(X))                (Rao-Blackwell theorem)

## Applications in Monte Carlo

- Replace noisy indicators `𝟙{X_i ∈ A}` with the analytic
  `P(A | Y_i)` where `Y_i` is a hidden variable you can integrate
  out.
- Gibbs sampling: replace `θ_i` draws with the closed-form
  conditional mean given the other parameters (Casella-Robert
  1996).
- Importance sampling: `E[f | proposal density]` instead of raw
  `f(X_i)`.

## Files

- `python/rao_blackwellization.py` — mixture-model example
  estimating `P(X > 2)` where `X = Z·A + (1−Z)·B` with `Z ~
  Bern(0.4)`, `A ~ N(0, 1)`, `B ~ N(3, 1)`. Plain MC via indicator
  vs Rao-Blackwell using analytic `P(X > 2 | Z)`. Demo (n=1000,
  500 reps): variance ratio MC / RB ≈ 1.5×.
- `r/rao_blackwellization.R` — no dedicated CRAN package; applied
  inside samplers (MCMCpack, rstan) as post-processing.

## When to use

- **You can integrate out a random component analytically** — Gibbs
  augmentation, mixture-membership indicators, latent-variable
  models.
- **Estimating tail probabilities / rare events** — averaging
  smooth conditional probabilities beats noisy indicators.
- **MCMC diagnostics / summaries** — report conditional posterior
  means rather than raw draws.

## When NOT to use

- **No tractable conditional** — the whole point is that
  `E[T | S]` is closed form.
- **Sufficiency doesn't help variance** — for some estimators the
  RB variant is identical (equal-variance case).
- **Computational cost of the conditional exceeds sample savings** —
  when the closed form is itself expensive (integrals, matrix
  inverses).

## Assumptions & caveats

- **Sufficient statistic exists** and is analytically tractable to
  condition on.
- **Estimator is a function of data** — RB requires `T = T(X)`,
  not a fixed constant.
- **Unbiasedness preserved** — `E[T*] = E[T]` by tower property.
- **MSE gain guaranteed but often modest** — a few percent to
  many-fold, depending on how much conditioning smooths.

## Related in this repo

- `monte-carlo-simulation`, `importance-sampling`, `nonparametric-bootstrap`
  — MC family.
- `gibbs-sampler`, `mcmc-metropolis-hastings` — canonical
  applications.
- `antithetic-control-variates`, `cuped-variance-reduction` —
  variance-reduction siblings.
- `semiparametric-efficiency`, `influence-functions-eif` — related
  theoretical variance-reduction.

## Run

```
python techniques/rao-blackwellization/python/rao_blackwellization.py
Rscript techniques/rao-blackwellization/r/rao_blackwellization.R
```

**Refs:** Rao, C.R. "Information and the accuracy attainable in the estimation of statistical parameters." *Bulletin of the Calcutta Mathematical Society*, 37: 81-91, 1945; Blackwell, D. "Conditional expectation and unbiased sequential estimation." *Ann. Math. Stat.*, 18: 105-110, 1947; Casella, G. & Robert, C.P. "Rao-Blackwellisation of sampling schemes." *Biometrika*, 83(1): 81-94, 1996.

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
