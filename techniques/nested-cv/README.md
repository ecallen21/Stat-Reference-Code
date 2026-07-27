# Nested Cross-Validation + Stratified Repeated CV (Reference §10.13)

## The problem plain CV doesn't solve

If you tune hyperparameters (regularization strength, tree depth, k in kNN) using K-fold CV and then report that same CV score as your model's performance, you have **optimistic bias**: the model was scored on data used to pick its own hyperparameters. The reported score reflects *hyperparameter fit to the CV folds*, not out-of-sample generalization.

## Nested CV

Two nested loops:

- **Outer loop** (`K_outer` folds): holds out a *test* set — the score on this set is the honest performance estimate.
- **Inner loop** (`K_inner` folds inside each outer training set): grid-search hyperparameters.
- For each outer fold, refit on all outer-training data with the winning hyperparameter; score on outer-test.
- Aggregate the K_outer test scores.

**Important**: the outer-loop scores estimate the *tuning-and-training procedure's* generalization performance, not any *specific* model. The final production model is a **separate refit** on the full dataset with the winning hyperparameter (or one picked by rerunning inner CV on all the data).

## Stratified repeated CV

Repeat K-fold `R` times with different random seeds and average. Reduces variance from the particular fold assignment. Stratification (preserving class proportions) is standard for classification with imbalanced classes.

## Files

- `python/nested_cv.py` — from-scratch nested CV driver and stratified repeated-CV helper. Ridge-regression demo picks a hyperparameter per outer fold; logistic-regression stratified-repeated example.
- `r/nested_cv.R` — from-scratch nested CV.

## Assumptions

- Same as plain CV: independent observations; no leakage between folds.
- Enough data per outer fold to make the inner CV meaningful. Rule of thumb: `n / K_outer / K_inner ≥ 10 * (# parameters)`.
- Compute cost: `K_outer × K_inner × |hp_grid|` model fits. Nested 5×3 with a 4-value grid = 60 fits.

## Run

```
python techniques/nested-cv/python/nested_cv.py
Rscript techniques/nested-cv/r/nested_cv.R
```

**Refs:** Varma, S. & Simon, R. "Bias in error estimation when using cross-validation for model selection." *BMC Bioinformatics* 7, 91, 2006; Cawley, G.C. & Talbot, N.L.C. "On over-fitting in model selection and subsequent selection bias in performance evaluation." *JMLR* 11, 2079–2107, 2010; Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch. 7).

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
