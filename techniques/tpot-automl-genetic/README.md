# TPOT — Tree-based Pipeline Optimization (Reference §47.244)

Olson & Moore (2016, ICML AutoML). Uses GENETIC PROGRAMMING to
search over full sklearn pipelines:

    individual = (preprocessor → feature-selection → … → estimator)
    fitness    = cross-validated score
    mutations  : swap components, tune hyperparameters
    crossover  : exchange sub-pipelines

After N generations, TPOT emits Python code for the best pipeline.
Slower than AutoGluon's stack but explores structural diversity.

## Files

- `python/tpot_automl_genetic.py` — toy GP search over
  (scaler × selector × estimator × k) on the diabetes dataset:
  - Best pipeline: **standard-scaler → SelectKBest → GBM**.
  - Best 3-fold CV MSE 3394 vs baseline ridge 3494 (**2.9 %
    improvement** from 4 generations × pop 6).
- `r/tpot_automl_genetic.R` — no R port; recommends `tpot`,
  `auto-sklearn`, `hyperopt-sklearn`.

## When to use

- **When you want to discover** the pipeline structure, not just
  hyperparameters.
- **Interpretable outcome** — TPOT emits Python code for the
  best pipeline.
- **Research / exploration** more than production serving.

## When NOT to use

- **Latency-critical AutoML** — GP is slow (many CV fits).
- **When AutoGluon's stack** already outperforms — often the
  case.
- **Very high-cardinality search spaces** — GP struggles beyond
  ~10 stages.

## Assumptions & caveats

- **Population size × generations** dictates search budget.
- **Mutation / crossover rates** — TPOT tunes these; defaults
  usually fine.
- **Fitness variance** — small folds → noisy scores → unstable
  winners.
- **Pareto front** (accuracy vs pipeline complexity) reported by
  TPOT.

## Related in this repo

- `autogluon-automl` — stacking-based sibling.
- `xgboost-boosting`, `lightgbm-histogram-boosting`,
  `catboost-ordered-boosting` — GBM neighbours.
- `bayesian-model-averaging`, `model-soups-fine-tune-averaging`
  — ensembling cousins.
- `genetic-algorithm`, `evolution-strategies-openai`,
  `cma-es-evolution-strategy` — evolutionary-optimisation
  ancestors.

## Run

```
python techniques/tpot-automl-genetic/python/tpot_automl_genetic.py
Rscript techniques/tpot-automl-genetic/r/tpot_automl_genetic.R
```

**Refs:** Olson, R. S. & Moore, J. H. "TPOT: A tree-based pipeline optimization tool for automating data science." *ICML AutoML Workshop*, 2016; Feurer, M. et al. "Auto-sklearn 2.0." *arXiv:2007.04074*, 2020.

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
