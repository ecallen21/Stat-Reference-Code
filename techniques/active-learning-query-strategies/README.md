# Active Learning -- Pool-Based Query Strategies (Reference §47.27)

Settles (2010, *Active Learning Literature Survey*). Iteratively
pick the most **informative** unlabelled example to query a human /
expensive oracle.

## Common strategies

| Strategy | Score to maximise |
|---|---|
| Uncertainty (least confidence) | `1 − max_c P(y = c | x)` |
| Margin | small `P(top1) − P(top2)` |
| Entropy | `H(P(y | x))` |
| QBC (query-by-committee) | committee disagreement |
| Expected model change / EMC | `‖∇_θ L(x, ŷ)‖` |
| Core-set / diversity | max distance from labelled |

## Files

- `python/active_learning_query_strategies.py` — random /
  uncertainty / margin loops on a small logistic classifier from
  scratch (with IRLS). Demo (2-D synthetic, B=40 queries):
  random 0.855, uncertainty 0.810, margin 0.810 — note that in
  BINARY classification uncertainty and margin coincide, and
  active learning does not always beat random.
- `r/active_learning_query_strategies.R` — `ALEval`,
  `activeselector` (R); `modAL`, `small-text`, `ALiPy` (Python).

## When to use

- **Expensive labelling** — clinical chart review, radiology
  markup, medical-QA labels.
- **Class-imbalanced problems** — active learning helps focus on
  the minority class.
- **High-dim / structured inputs** — uncertainty over embeddings
  can guide annotators.

## When NOT to use

- **Cheap labels** — random or full labelling is simpler.
- **Bootstrap / warm-up too small** — initial model bias steers
  queries poorly.
- **Concept drift** — actively sampled labels can lag the drift.

## Assumptions & caveats

- **Sampling bias** — active queries create non-iid training data;
  standard test-set evaluation still works but confidence
  intervals for models are messier.
- **Cold-start** — start with a random init pool (~10-100 labels)
  before switching to active queries.
- **Batch queries** — most oracles want batches; diversity /
  clustering enforces spread within a batch.
- **Stopping criterion** — pre-specify a label budget or a
  performance plateau.

## Related in this repo

- `covariate-shift-adaptation`, `concept-drift-adwin` — related
  data-shift topics.
- `selective-prediction`, `conformal-prediction`,
  `ood-detection` — uncertainty-based cousins.
- `bayesian-optimization`, `multi-armed-bandits` — sequential
  decision-making siblings.

## Run

```
python techniques/active-learning-query-strategies/python/active_learning_query_strategies.py
Rscript techniques/active-learning-query-strategies/r/active_learning_query_strategies.R
```

**Refs:** Settles, B. *Active Learning Literature Survey*, University of Wisconsin CS TR 1648, 2010; Cohn, D., Atlas, L. & Ladner, R. "Improving generalization with active learning." *Machine Learning*, 15(2): 201-221, 1994.

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
