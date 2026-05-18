# Binary Logistic Regression (Reference §7.1)

Model the log-odds of a binary outcome as a linear function of predictors:

`logit(πᵢ) = log(πᵢ / (1 − πᵢ)) = x'ᵢβ`,   `πᵢ = P(yᵢ = 1 | xᵢ)`

## Fitting via IRLS (Fisher scoring)

Each iteration is a weighted least-squares step on a "working response":

- `ηᵢ = x'ᵢβ`, `πᵢ = 1 / (1 + e^{−ηᵢ})`
- Weights: `wᵢ = πᵢ(1 − πᵢ)`
- Working response: `zᵢ = ηᵢ + (yᵢ − πᵢ)/wᵢ`
- Update: `β ← (X'WX)⁻¹ X'Wz`

Converges quadratically near the MLE. The inverse Fisher information `(X'WX)⁻¹` at convergence gives `Var(β̂)`.

## Output you get

| Quantity | Meaning |
|----------|---------|
| `βⱼ` | Change in log-odds per unit `xⱼ` |
| `exp(βⱼ)` | **Odds ratio** for `xⱼ` |
| `SE(βⱼ)` | From `√(diag (X'WX)⁻¹)` |
| Wald `z = β/SE` | Per-coef significance |
| Deviance | `−2 ll(β̂)`; smaller is better |
| LR `χ²` | `2(ll_full − ll_null)`, `df = p − 1` |
| AIC | `2p − 2 ll_full` |
| McFadden's `R²` | `1 − ll_full / ll_null` |

## Caveats / things to watch

- **Separation.** If a predictor perfectly separates the classes, the MLE diverges (`β̂ → ∞`). Symptoms: huge coefficients, huge SEs, "did not converge" warnings. Fix: **Firth's penalized MLE** (`techniques/firth-logistic`) or regularized logistic.
- **Wald CIs on `β` are symmetric**; `exp(±)` to get OR CIs is the standard, but profile-likelihood CIs (`techniques/wald-lrt-score`) are more accurate when the SE is large.
- **Risk ratios ≠ odds ratios.** For common outcomes (e.g. prevalence > 10%), the OR overstates the RR. Use **log-binomial** or **modified Poisson** (`techniques/modified-poisson`) when you specifically want an RR.
- **AUC / accuracy on training data ≠ honest performance.** Use cross-validation or a held-out set for evaluation.

## Files
- `python/logistic_regression.py` — from-scratch IRLS with full summary (coefficients, ORs + 95% CIs, deviance, LR test, AIC, McFadden R², accuracy); matches `statsmodels.Logit` to 12 decimals.
- `r/logistic_regression.R` — from-scratch IRLS + base `glm(family = binomial)`.
- `pyspark/logistic_regression.py` — `pyspark.ml.classification.LogisticRegression`; reports training accuracy + AUC.

## Run
```
python techniques/logistic-regression/python/logistic_regression.py
Rscript techniques/logistic-regression/r/logistic_regression.R
python techniques/logistic-regression/pyspark/logistic_regression.py
```

**Refs:** Hosmer, Lemeshow & Sturdivant, *Applied Logistic Regression*, 3rd ed., Wiley, 2013; McCullagh & Nelder, *Generalized Linear Models*, 2nd ed., Chapman & Hall, 1989.

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
