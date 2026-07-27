# Penalized Cox Regression: L1 / L2 / Elastic Net (Reference §11.21)

For high-dimensional Cox (`p` comparable to or larger than `n_events`), the ordinary Cox MLE is unstable or non-existent. **Penalized Cox** shrinks coefficients toward zero via a regularization term added to the negative partial log-likelihood:

```
minimize  −log PL(β)  +  λ · [ (1 − α) · ‖β‖₂² / 2  +  α · ‖β‖₁ ]

α = 0    → ridge (L2)                       stability under collinearity
α = 1    → lasso (L1)                       automatic variable selection
0 < α < 1 → elastic net                     selection + grouping of correlated predictors
```

## Algorithm

**Coordinate descent** (Simon, Friedman, Hastie & Tibshirani 2011): at each iteration, take a Newton-Raphson step decomposition of the partial log-likelihood; update each `β_j` in turn via soft-thresholding for the L1 part and a shrinkage denominator for the L2 part.

## Choosing λ

Use **cross-validated partial log-likelihood** (`cv.glmnet(family="cox")` in R). Two standard choices:

- `λ_min`: the λ minimizing CV error. Lowest CV loss, largest model.
- `λ_1se`: the largest λ within 1 SE of `λ_min`. Simpler model, similar performance.

## When to use

- `p > n_events` (biomarker / genomics survival studies).
- Highly collinear predictors — ridge or elastic net stabilize.
- Want a sparse subset of predictors — lasso or elastic net.

## Files

- `python/penalized_cox.py` — pedagogical coordinate-descent solver for elastic-net Cox + regularization path. On synthetic data with true nonzeros = `[0, 1, 2]`, lasso at moderate λ correctly shrinks most non-important coefficients toward zero.
- `r/penalized_cox.R` — thin wrapper around `glmnet(family = "cox")`, which is the authoritative implementation and includes cross-validated λ selection.

## Assumptions

- Same as ordinary Cox: proportional hazards, independent censoring.
- **Standardization matters** — the penalty is applied on the standardized scale so predictor units don't dominate the penalty. Predictors are auto-standardized inside `penalized_cox()`.
- CV for λ should respect any clustering structure in the data.

## Run

```
python techniques/penalized-cox/python/penalized_cox.py
Rscript techniques/penalized-cox/r/penalized_cox.R
```

**Refs:** Tibshirani, R. "The lasso method for variable selection in the Cox model." *Stat. Med.* 16(4), 385–395, 1997; Simon, N., Friedman, J., Hastie, T. & Tibshirani, R. "Regularization paths for Cox's proportional hazards model via coordinate descent." *J. Stat. Soft.* 39(5), 1–13, 2011; Verweij, P.J.M. & van Houwelingen, H.C. "Penalized likelihood in Cox regression." *Stat. Med.* 13(23–24), 2427–2436, 1994.

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
