# Regularization: Ridge, Lasso, Elastic Net (Reference §5.9, §5.17)

All three add a penalty on the size of the coefficients to the OLS objective:

`minimize  RSS(β) + λ · Penalty(β)`

| Method | Penalty | Behavior |
|--------|---------|----------|
| **Ridge** (L2) | `Σ β_j²` | Shrinks toward 0; **never exactly zero**; handles collinearity gracefully |
| **Lasso** (L1) | `Σ \|β_j\|` | Shrinks AND **drops** variables (sparse solutions); arbitrarily picks one of a correlated group |
| **Elastic net** | `α·Σ\|β_j\| + (1−α)/2 · Σβ_j²` | Mix: keeps correlated groups together; `α=1` lasso, `α=0` ridge |

## Why standardize first

The penalty acts on raw coefficients. If `x₁` is in meters and `x₂` in kilometers, they take very different "amounts of penalty" for the same effect. Standard practice (followed throughout this file): **center and scale every column** before fitting, then back-transform. R's `glmnet` and sklearn's `Ridge`/`Lasso` standardize by default.

## Algorithms

- **Ridge** has a closed form: `β̂ = (X'X + λI)⁻¹ X'y`. Cheap.
- **Lasso** has **no closed form** because of the `|·|` corners. Standard solver: **cyclic coordinate descent with soft-thresholding** — for each coordinate, the update is

  `β_j ← S(z_j, λ/n)`,  where `z_j = ⟨x_j, r⟩/n` (residual without the `j`-th contribution) and `S(z, t) = sign(z) · max(0, |z|−t)` is the soft-threshold.

  Converges in `O(np · iter)`; iter is typically a few dozen.
- **Elastic net** uses the same loop with `β_j ← S(z_j, λα/n) / (1 + λ(1−α)/n)`.

## Choosing λ

Cross-validation (`cv_regularization` here): fit on `k − 1` folds, evaluate MSE on the held-out fold, average across folds. The CV-min λ is the standard choice; the "1-SE" rule picks the largest λ within 1 SE of the min (more parsimonious).

## When to use which

- **Pure prediction, all predictors potentially relevant** → ridge.
- **Pure prediction, many irrelevant predictors expected** → lasso (variable selection embedded).
- **Many correlated predictors and you want groups in/out together** → elastic net.
- **You want valid inference (CIs / p-values)** → all three are biased and getting calibrated inference is a research problem (post-selection inference, debiased lasso). For inference, prefer plain OLS with deliberate variable choice.

## Files
- `python/regularization.py` — from-scratch ridge (closed form), lasso (coordinate descent with soft-thresholding), elastic net (same loop with EN update), and `cv_regularization` for k-fold CV; compares against `sklearn.linear_model` Ridge / Lasso / ElasticNet.
- `r/regularization.R` — from-scratch versions + `glmnet::glmnet` and `glmnet::cv.glmnet` (the canonical implementation).
- `pyspark/regularization.py` — `pyspark.ml.regression.LinearRegression` with `regParam = λ` and `elasticNetParam = α`; ridge / lasso / elastic-net all dispatch from one model class.

## Run
```
python techniques/regularization/python/regularization.py
Rscript techniques/regularization/r/regularization.R
python techniques/regularization/pyspark/regularization.py
```

**Refs:** Hoerl & Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems," *Technometrics* 12(1), 55–67, 1970; Tibshirani, "Regression Shrinkage and Selection via the Lasso," *JRSS-B* 58(1), 267–288, 1996; Zou & Hastie, "Regularization and Variable Selection via the Elastic Net," *JRSS-B* 67(2), 301–320, 2005; Friedman, Hastie & Tibshirani, "Regularization Paths for Generalized Linear Models via Coordinate Descent," *J. Stat. Software* 33(1), 2010 (the `glmnet` paper).

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
