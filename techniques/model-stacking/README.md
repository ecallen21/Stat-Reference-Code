# Model Stacking / Super Learner (Reference §26.14)

Combine predictions of several **base learners** via a **meta learner**. To avoid overfitting, meta features must come from **out-of-fold (OOF)** predictions.

## Recipe

```
1. Split training data into K folds.
2. For each base model b and fold k:
      train on (folds != k), predict on fold k -> OOF prediction.
3. Train META model on OOF predictions -> y.
4. Test time: fit each base on ALL training data, feed test predictions to meta.
```

Same idea as van der Laan et al.'s **Super Learner** with an ensemble of arbitrary base learners.

## Typical setup

- **Diverse base learners** — one linear, one tree-based, one kNN, one boosted. Diversity matters more than raw accuracy.
- **Simple meta** — ridge regression or logistic (avoid a base that already overfits meta features).

## Files

- `python/model_stacking.py` — generic K-fold stacking with pluggable base + meta functions. Demo (linear + tree + kNN, ridge meta): stacked OOF RMSE 0.319 beats every individual base (linear 0.49, tree 0.40, kNN 0.32).
- `r/model_stacking.R` — `SuperLearner::SuperLearner(SL.library = c("SL.lm", "SL.rpart", "SL.knn"))` or the tidymodels `stacks` package.

## When to use

- **Squeezing the last few % of accuracy** — Kaggle winners routinely stack.
- **Combining models with complementary biases** — linear for the smooth part, trees for the nonlinear part.
- **Ensembling ML pipelines** built by different teams.

## Assumptions & caveats

- **OOF is essential** — training the meta on in-sample base predictions leaks and overfits.
- **Meta learner regularization** — use ridge or LASSO to avoid overfitting the base predictions.
- **Marginal improvement** — typically 1–3% over the best single base; more when bases are more diverse.
- **Prediction cost** — proportional to sum of base costs.

## Related

- **Bagging** — average predictions from bootstrap-fit copies of one learner (see `random-forest`).
- **Boosting** — sequential; each learner corrects the previous.
- **Blending** — like stacking but with a single holdout rather than K-fold; simpler and slightly more overfitting-prone.

## Run

```
python techniques/model-stacking/python/model_stacking.py
Rscript techniques/model-stacking/r/model_stacking.R
```

**Refs:** Wolpert, D.H. "Stacked generalization." *Neural Netw.* 5(2), 241–259, 1992; Breiman, L. "Stacked regressions." *Mach. Learn.* 24(1), 49–64, 1996; van der Laan, M.J., Polley, E.C. & Hubbard, A.E. "Super Learner." *Stat. Appl. Genet. Mol. Biol.* 6(1), 25, 2007.

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
