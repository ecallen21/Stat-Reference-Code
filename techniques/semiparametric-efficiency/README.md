# Semiparametric Efficiency (Reference §33.4)

For a target parameter `θ` with **nuisance parameters** (a propensity
score, a mean model, a density shape), the **semiparametric efficiency
bound** is the Cramér-Rao lower bound over all regular estimators:

```
Var(θ̂)  ≥  (1 / n) · 𝔼[ φ(O)² ]
```

`φ(O)` is the **efficient influence function** (EIF) at the truth
(Bickel-Klaassen-Ritov-Wellner 1993; Tsiatis 2006). Any estimator whose
influence function equals `φ` is **asymptotically efficient**.

## Canonical illustration: mean under missingness

Data `(X, R, Y)` with `Y` observed only when `R = 1`. Target
`θ = 𝔼[Y]`.

- **IPW** — inverse-probability-weighted:
  `θ̂_IPW = mean( R · Y / π̂(X) )`.
- **AIPW** (a.k.a. doubly-robust) — augments with a mean-model term:
  `θ̂_AIPW = mean( μ̂(X) + R · (Y − μ̂(X)) / π̂(X) )`.

AIPW attains the semiparametric-efficiency bound; IPW does not.

## When to use

- **Causal inference** (ATE, ATT), missing-data mean estimation,
  survival TMLE.
- **You want a valid asymptotic variance** without over-specifying a
  parametric model.
- **Double-robustness** — AIPW is consistent if EITHER the propensity
  OR the mean model is correctly specified.

## When NOT to use

- **Nuisance models are all wrong** — no efficiency theorem saves you.
- **Small n** — plug-in bias in nuisance models dominates the finite-
  sample MSE; cross-fitting (`double-ml`) helps.

## Files

- `python/semiparametric_efficiency.py` — Monte-Carlo comparison of
  `IPW` vs `AIPW` on a synthetic missing-outcome problem (500 trials
  of n=500). Variance ratio `IPW / AIPW ≈ 1.18` — AIPW is 18 % more
  efficient, empirically matching the theoretical bound.
- `r/semiparametric_efficiency.R` — `tmle` / `drtmle` / `npcausal`
  (R); `econml`, `dowhy`, `zEpid` (Python).

## Assumptions & caveats

- **Nuisance quality** — the EIF only works when nuisance models are
  fit with `sqrt(n)`-rate quality; use flexible learners + cross-fitting
  (Chernozhukov 2018).
- **Positivity** — propensities near 0 or 1 make IPW / AIPW variance
  explode; trim or use overlap-weighted estimators.
- **Efficiency vs robustness** — AIPW gains efficiency by combining
  models; a single-badly-specified nuisance still returns bias.
- **EIF derivation** — for complex parameters requires functional-
  derivative computation (`influence-functions-eif`).

## Related in this repo

- `influence-functions-eif` — the object that defines efficient
  estimators.
- `tmle-doubly-robust`, `propensity-score-methods` (if present) —
  causal-inference cousins.
- `covariate-shift-adaptation` — density-ratio IPW under shift.
- `sandwich-robust-se` — SE that captures the influence function via
  Huber-White.
- `double-ml` (if present) — cross-fitting for efficient nuisance
  learners.

## Run

```
python techniques/semiparametric-efficiency/python/semiparametric_efficiency.py
Rscript techniques/semiparametric-efficiency/r/semiparametric_efficiency.R
```

**Refs:** Bickel, P.J., Klaassen, C.A.J., Ritov, Y. & Wellner, J.A. *Efficient and Adaptive Estimation for Semiparametric Models*, Johns Hopkins University Press, 1993; Tsiatis, A. *Semiparametric Theory and Missing Data*, Springer, 2006; Chernozhukov, V. et al. "Double/debiased machine learning." *Econometrics Journal*, 2018.

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
