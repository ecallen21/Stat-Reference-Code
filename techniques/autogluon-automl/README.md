# AutoGluon-style AutoML (Reference §47.243)

Erickson et al. (2020). Zero-hyperparameter AutoML for tabular
data:

1. Train MANY diverse models (GBM variants, RF, NN, linear).
2. **STACK** them: level-1 models feed predictions into a
   level-2 meta-learner (weighted average or another GBM).
3. **Bagging + k-fold** ensembling.

Consistently top-ranks on Kaggle / OpenML with a single `fit()`
call. Sibling: FLAML (Wang 2021, cost-frugal AutoML).

## Files

- `python/autogluon_automl.py` — 4-model stack (ridge / RF / GBM
  / KNN) on the diabetes dataset:
  - Individual base-model MSEs shown.
  - Uniform-weighted stack: 3303.7.
  - Fitted stack weights (NNLS on val): 0.26 ridge, 0.08 RF,
    0.43 GBM, 0.23 KNN.
- `r/autogluon_automl.R` — no R port; recommends
  `autogluon-tabular`, `flaml`, `h2o.automl`.

## When to use

- **Tabular ML competition / production** where model choice is
  hard.
- **When compute budget is generous** — trains many models.
- **Ensemble baseline** before hand-tuning specific models.

## When NOT to use

- **Latency-critical inference** — an ensemble adds many
  forward passes.
- **Very interpretable models required** — stacked ensembles are
  opaque.
- **When a single well-tuned GBM already wins** on your data.

## Assumptions & caveats

- **Data cleanliness** helps but isn't required — AutoGluon has
  auto-encoders for missing / categorical / text.
- **Stack weights** via NNLS or level-2 GBM; both work.
- **Bagging + k-fold** reduces variance further than a single
  stack.
- **Time / memory limits** control model-family enable list.

## Related in this repo

- `bayesian-model-averaging`, `model-soups-fine-tune-averaging`,
  `deep-ensembles` — ensembling cousins.
- `tpot-automl-genetic` — genetic-programming alternative.
- `xgboost-boosting`, `lightgbm-histogram-boosting`,
  `catboost-ordered-boosting` — base-model neighbours.

## Run

```
python techniques/autogluon-automl/python/autogluon_automl.py
Rscript techniques/autogluon-automl/r/autogluon_automl.R
```

**Refs:** Erickson, N. et al. "AutoGluon-Tabular: Robust and accurate AutoML for structured data." *arXiv:2003.06505*, 2020; Wang, C., Wu, Q., Weimer, M. & Zhu, E. "FLAML: A fast and lightweight AutoML library." *MLSys*, 2021.

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
