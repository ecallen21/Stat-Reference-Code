# Information Criteria: AIC / BIC / AICc / DIC / WAIC (Reference §34.5)

**Model selection criteria** that penalise complexity by adding a
penalty to −2 log-likelihood.

## Formulas

| Criterion | Formula                                                     | Target |
|----------:|:------------------------------------------------------------|:-------|
| **AIC**   | `−2 log L + 2 k`                                             | Prediction (K-L divergence to truth) |
| **AICc**  | `AIC + 2k(k+1)/(n − k − 1)`                                  | Prediction (small n) |
| **BIC**   | `−2 log L + k log n`                                         | Model recovery (posterior odds) |
| **DIC**   | `−2 log L(θ̄) + 2 p_D`                                       | Bayesian prediction (posterior samples) |
| **WAIC**  | `−2 (lppd − p_WAIC)`                                          | Bayesian prediction (out-of-sample) |

Lower is better in all cases. AIC and BIC differ philosophically: AIC
targets predictive accuracy; BIC targets recovery of the true model.

## When to use

- **Model selection among a small candidate set** — nested / non-nested.
- **BIC** for scientific model identification (asymptotically
  consistent when the truth is in the model set).
- **AIC / AICc** for prediction; AICc when `n / k < 40`.
- **WAIC / LOO** for Bayesian model comparison (proper Bayesian
  out-of-sample estimator).

## When NOT to use

- **Non-comparable models** (different response transformations).
- **Cross-validation is preferred** if you have compute for it.
- **DIC** has known issues; prefer WAIC/LOO for modern Bayesian
  workflows.

## Files

- `python/information_criteria.py` — from-scratch OLS log-likelihood
  + AIC / AICc / BIC over polynomial orders 1-7 on synthetic cubic
  truth. **AIC picks order 4** (over-fit); **BIC picks order 3**
  (truth); AICc close to AIC.
- `r/information_criteria.R` — `stats::AIC/BIC`, `MuMIn::AICc`,
  `loo::waic/loo` (R); `statsmodels`, `pymc`, `arviz` (Python).

## Assumptions & caveats

- **Same likelihood family** across models being compared.
- **AIC vs BIC** — different penalties, different targets. Report both.
- **AICc** — needed when `n / k ≲ 40`.
- **DIC / WAIC / LOO** — assume the posterior is well-sampled.
- **Non-nested comparisons** — Vuong tests, Bayes factors, or
  cross-validation.

## Related in this repo

- `cross-validation` — the gold standard for prediction accuracy.
- `bayesian-model-averaging`, `bayesian-model-comparison` — Bayesian
  alternatives.
- `stepwise-regression` (if present) — commonly uses AIC / BIC.
- `bayes-factor` (adjacent) — related Bayesian model selection.

## Run

```
python techniques/information-criteria/python/information_criteria.py
Rscript techniques/information-criteria/r/information_criteria.R
```

**Refs:** Akaike, H. "A new look at the statistical model identification." *IEEE TAC*, 1974; Schwarz, G. "Estimating the dimension of a model." *Annals of Statistics*, 1978; Watanabe, S. "Asymptotic equivalence of Bayes cross validation and widely applicable information criterion." *JMLR*, 2010; Spiegelhalter, D.J. et al. "Bayesian measures of model complexity and fit (DIC)." *JRSS-B*, 2002.

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
