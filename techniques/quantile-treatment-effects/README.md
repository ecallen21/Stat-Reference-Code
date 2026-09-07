# Quantile Treatment Effects (Reference §15.33)

Firpo (2007). The QTE at quantile τ ∈ (0, 1) is:

    QTE(τ) = Q_τ[Y(1)] − Q_τ[Y(0)]

Unlike the ATE (a single mean-shift number), QTE reveals **how the
treatment effect varies along the outcome distribution** — e.g. a
programme that helps median earners while hurting the bottom quintile.

## Firpo IPW estimator

Under unconfoundedness, invert the weighted empirical CDF:

    F̂_1(y) = Σ Tᵢ·1[Yᵢ ≤ y] / e(Xᵢ)   /   Σ Tᵢ / e(Xᵢ)
    F̂_0(y) = Σ (1−Tᵢ)·1[Yᵢ ≤ y] / (1−e(Xᵢ))   /   Σ (1−Tᵢ) / (1−e(Xᵢ))
    Q̂_τ(t) = inf { y : F̂_t(y) ≥ τ }

## Files

- `python/quantile_treatment_effects.py` — logistic PS + Firpo IPW
  quantile estimator. Demo (n=6000, rank-shift τ_i = 0.5 + 2·rank):
  QTE(0.10)=+0.63 (truth 0.70), QTE(0.50)=+1.44 (truth 1.50),
  QTE(0.90)=+2.31 (truth 2.30) — the growing gap invisible to an ATE.
- `r/quantile_treatment_effects.R` — `Counterfactual::counterfactual`,
  `quantreg::rq`, `qte` (R); `econml.dr.LinearDRLearner`,
  `causalml.metalearners` (Python).

## When to use

- **Heterogeneous / distributional effects** — you suspect
  treatment effects vary across the outcome distribution.
- **Policy analysis** — inequality-relevant impacts (minimum wage,
  cash transfers, education programmes).
- **Skewed outcomes** — medical costs, wait times, biomarkers where
  the mean is a poor summary.

## When NOT to use

- **Small n** — quantile estimators need many observations in each
  tail; rule of thumb ≥100 in each treatment arm at each τ of
  interest.
- **Rare outcomes** — QTE for binary or count outcomes reduces to
  differences of proportions; use risk-difference/OR instead.
- **You only care about the mean** — ATE (IPTW, AIPW) is simpler and
  more efficient.

## Assumptions & caveats

- **Unconfoundedness** (`Y(t) ⊥ T | X`) and **positivity** — same as
  IPW.
- **Common quantile support** — Q̂_τ is undefined outside the observed
  range; extrapolation is dangerous.
- **Not rank preservation** — QTE compares marginals, not the same
  unit's Y(1) vs Y(0); do not interpret as "the τ-th person's
  effect".
- **SE** — bootstrap (the pair or wild bootstrap); analytic SEs
  (Firpo 2007) require the density estimator to be accurate.

## Related in this repo

- `iptw`, `overlap-weighting` — mean-based counterparts.
- `quantile-regression` — conditional (regression) QTE.
- `aipw-doubly-robust` — doubly-robust version via influence functions.
- `bayesian-quantile-regression` — Bayesian alternative.

## Run

```
python techniques/quantile-treatment-effects/python/quantile_treatment_effects.py
Rscript techniques/quantile-treatment-effects/r/quantile_treatment_effects.R
```

**Refs:** Firpo, S. "Efficient semiparametric estimation of quantile treatment effects." *Econometrica*, 75(1): 259-276, 2007; Chernozhukov, V. & Hansen, C. "An IV model of quantile treatment effects." *Econometrica*, 2005.

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
