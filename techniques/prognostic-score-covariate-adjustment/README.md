# Prognostic-Score Covariate Adjustment / PROCOVA (Reference §44.15)

Hansen (2008); Schuler et al. (2022, PROCOVA). In a randomised
trial, adjusting the outcome for a **model-based prognostic score**
built from control-group outcomes reduces variance of the treatment
effect estimate.

## Recipe

1. Fit `m̂(x) = E[Y_control | X]` on external / historical data
   (typically ML on observational or trial control-arm data).
2. ANCOVA: `Y = β₀ + β₁ · T + β₂ · m̂(X) + ε`. `β̂₁` is the
   adjusted ATE.

## Efficiency

    Var(τ̂_adj) / Var(τ̂_unadj)  ≈  1 − R²(m̂)

A prognostic score with `R² = 0.5` **halves** the variance ⇒ 2×
effective sample size. `R² = 0.66` ⇒ 3× gain.

**Regulatory status:** EMA CHMP (2024) has issued a qualification
opinion allowing PROCOVA in regulatory clinical-trial submissions.

## Files

- `python/prognostic_score_covariate_adjustment.py` — historical
  OLS prognostic model + ANCOVA with sandwich SE from scratch.
  Demo (n_hist=5000, trial n=400, p=5, τ=0.60): unadjusted τ̂=0.624
  (SE 0.178); PROCOVA τ̂=0.665 (SE 0.103); variance ratio 0.335 →
  3× ESS gain.
- `r/prognostic_score_covariate_adjustment.R` — `RATES`
  (Unlearn.AI PROCOVA), `MatchIt + prognosticscores`, `optmatch`,
  `survival` (R); `procova`, from-scratch (Python).

## When to use

- **Randomised trials with rich historical control data** —
  oncology, Alzheimer's, orphan diseases where external control
  cohorts exist.
- **Endpoints with known predictors** — vitals, biomarkers,
  imaging.
- **Variance reduction / sample-size saving** — trade compute for
  fewer patients.
- **Regulatory-grade covariate adjustment** — EMA qualification;
  FDA discussions ongoing.

## When NOT to use

- **Prognostic model overfits** to historical data — cross-validate
  and freeze the model before the trial.
- **Concurrent covariate model chosen post-hoc** — undermines
  pre-specification; PROCOVA requires the prognostic score be
  frozen ex ante.
- **Very small trials with weak external data** — the variance
  reduction shrinks; simpler stratification may be better.

## Assumptions & caveats

- **Randomisation is intact** — `T ⊥ X` in expectation; the score
  is a covariate, not a confounder-adjuster.
- **Prognostic score built on control outcome only** — do not
  include treatment or post-treatment variables.
- **Pre-specify and lock the model** before analysing trial data.
- **Nonlinear m̂** is fine — the ANCOVA linearly adjusts for the
  (possibly nonlinear) score, keeping the ATE unbiased.

## Related in this repo

- `mde-sample-size`, `cuped-variance-reduction` — variance-reduction
  cousins.
- `iptw`, `aipw-doubly-robust` — observational-adjustment
  techniques.
- `external-validation`, `clinical-risk-scores` — model-development
  siblings.

## Run

```
python techniques/prognostic-score-covariate-adjustment/python/prognostic_score_covariate_adjustment.py
Rscript techniques/prognostic-score-covariate-adjustment/r/prognostic_score_covariate_adjustment.R
```

**Refs:** Hansen, B.B. "The prognostic analogue of the propensity score." *Biometrika*, 95(2): 481-488, 2008; Schuler, A. et al. "Increasing the efficiency of randomized trial estimates via linear adjustment for a prognostic score." *Int J Biostat*, 18(2): 329-356, 2022; EMA CHMP qualification opinion for PROCOVA, 2024.

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
