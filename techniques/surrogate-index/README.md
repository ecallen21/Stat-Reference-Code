# Surrogate Index for Long-Term Effects (Reference §44.14)

Athey-Chetty-Imbens-Kang (2020), Hohnhold-O'Brien-Tang (2015).
Combine **short-term proxies** into an index that predicts the
**long-term outcome** on a historical cohort, then use the index
as a stand-in for the long-term outcome in a current experiment
where only short-term signals are observed.

## Two-stage estimation

1. **Historical fit** — regress long-term Y on short-term S:
   `Y = α + Σ β_j · S_j + ε` → save `(α, β)`.
2. **Experiment analysis** — construct
   `Y_index = α + S_current β̂` and analyse the treatment effect on
   `Y_index`. SEs must be adjusted for the estimated weights
   (two-stage / bootstrap).

## When to use

- **Product experiments** where the outcome you care about takes
  weeks/months to observe (retention, LTV, churn).
- **Rapid decision-making** — need a signal well before the full
  outcome is measurable.

## When NOT to use

- **No historical population with both S and Y observed** — no
  weights to fit.
- **Non-stationary surrogates** — the S→Y relationship must be
  stable between historical and experimental cohorts.

## Files

- `python/surrogate_index.py` — historical `S → Y` OLS
  regression + surrogate-index construction in the experiment
  cohort. Demo (n_hist=5000, n_exp=2000, 3 short-term signals):
  surrogate-index treatment effect **+0.182** vs true long-term
  effect ~0.197 (`Σ β · 0.20`).
- `r/surrogate_index.R` — `stats::lm`, `mediation` (R);
  `sklearn.LinearRegression`, custom (Python).

## Assumptions & caveats

- **Surrogacy condition** (Prentice) — S must fully mediate the
  treatment effect on Y for the index to be valid.
- **Validate on a holdout historical experiment** where both S and
  Y are observed and the treatment ran to completion.
- **Two-stage SEs** — plug-in inference is under-conservative;
  bootstrap or asymptotic corrections needed.
- **Report both** the index estimate and, when available, direct
  long-term measurements.

## Related in this repo

- `mediation-analysis` (if present) — surrogacy is a mediation
  concept.
- `long-term-outcomes` (if present) — companion.
- `ab-test-fundamentals`, `cuped-variance-reduction` — the
  short-term analytics.

## Run

```
python techniques/surrogate-index/python/surrogate_index.py
Rscript techniques/surrogate-index/r/surrogate_index.R
```

**Refs:** Athey, S., Chetty, R., Imbens, G.W., & Kang, H. "The surrogate index: combining short-term proxies to estimate long-term treatment effects more rapidly and precisely." *NBER Working Paper*, 2020; Hohnhold, H., O'Brien, D., & Tang, D. "Focusing on the long-term: it's good for users and business." *KDD*, 2015.

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
