# AdaBoost Classifier (Reference §47.69)

Freund & Schapire (1997). Adaptive Boosting: sequentially train
weak learners on reweighted data,

    ε_t = Σ D_t(i) · 𝟙{h_t(x_i) ≠ y_i},   α_t = ½ log((1−ε_t)/ε_t),
    D_{t+1}(i) ∝ D_t(i) · exp(−α_t · y_i · h_t(x_i)),
    F(x) = sign( Σ_t α_t · h_t(x) )   (y ∈ {−1, +1}).

Equivalent to forward stagewise additive minimisation of the
exponential loss (Friedman-Hastie-Tibshirani 2000).

## Files

- `python/adaboost_classifier.py` — from-scratch AdaBoost.M1
  with sklearn decision-stump base learners + sample_weight. Demo
  (concentric-ring target with 5% label flip):
  - Single stump train acc = 0.635
  - Depth-3 tree train acc = 0.828
  - AdaBoost T=10  train acc = 0.901
  - AdaBoost T=200 train acc = 0.914.
- `r/adaboost_classifier.R` — `adabag::boosting`,
  `fastAdaboost` (R); `sklearn.ensemble.AdaBoostClassifier`,
  from-scratch (Python).

## When to use

- **Weak-learner ensembling** — stumps for interpretability, or
  shallow trees.
- **Class imbalance** — reweighting focuses on misclassified minority.
- **When gradient-boosting or bagging is too complex** — AdaBoost
  is a small, well-understood algorithm.

## When NOT to use

- **Very noisy labels** — AdaBoost concentrates weight on hard
  examples; label noise fatal (mitigate with LogitBoost or
  Gentle AdaBoost).
- **Complex nonlinear boundaries at scale** — gradient boosting
  (XGBoost, LightGBM, CatBoost) usually dominates.
- **Regression** — use gradient boosting or LSBoost variants.
- **Highly imbalanced with rare positives + strict AUC** — try
  cost-sensitive or focal-loss trees.

## Assumptions & caveats

- **Weak learners ε_t < 0.5** — else the loop stops.
- **Exponential loss** — sensitive to outliers; try LogitBoost /
  Real AdaBoost for smoother alternatives.
- **Overfitting** — bounded by margin theory (Schapire et al 1998);
  still, monitor test error.
- **Randomness** — result reproducible given stump splits + weights;
  no bootstrap involved.

## Related in this repo

- `gradient-boosting`, `bart-bayesian-additive-regression-trees`,
  `bagging-oob`, `random-forest` — tree-ensemble family.
- `decision-tree` — the weak learner.
- `deep-ensembles`, `mc-dropout` — modern uncertainty ensembles.
- `explainable-boosting-machine`, `shap-values`,
  `shap-interactions` — interpretable-boosting cousins.

## Run

```
python techniques/adaboost-classifier/python/adaboost_classifier.py
Rscript techniques/adaboost-classifier/r/adaboost_classifier.R
```

**Refs:** Freund, Y. & Schapire, R.E. "A decision-theoretic generalization of on-line learning and an application to boosting." *JCSS* 55(1): 119-139, 1997; Friedman, J., Hastie, T. & Tibshirani, R. "Additive logistic regression: a statistical view of boosting." *Ann Stat* 28(2): 337-407, 2000.

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
