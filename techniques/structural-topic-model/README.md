# Structural Topic Model (Reference §42.7)

Roberts-Stewart-Tingley (2019). Extension of LDA that lets
**document-level covariates** influence both:

- **Topic prevalence** — the probability of using a topic (document-
  topic distribution).
- **Topic content** — word choice within a topic (topic-word
  distribution).

Bridges topic modelling and regression: after fitting, use
`estimateEffect` to test whether covariates shift topic prevalence.

## When to use

- **Text + metadata** — political speeches with party, survey
  responses with respondent demographics, clinical notes with
  visit type.
- **Effect-of-covariate on discourse** — replaces ad-hoc "topic
  proportion regression" with a principled model.

## When NOT to use

- **No covariates** — plain LDA is simpler.
- **Very small corpora** — STM prevalence estimates are unstable
  below a few hundred documents.

## Files

- `python/structural_topic_model.py` — toy EM STM with group-
  specific Dirichlet prevalence prior. Demo (60 documents, 6-word
  vocab, 2 topics: medical and sport, 2 publication venues): recovers
  **top-3 words per topic correctly** (health/doctor/hospital vs
  game/sport/team) and **per-group prevalence 0.84 / 0.16 vs
  0.18 / 0.82** — venue effect on topic use.
- `r/structural_topic_model.R` — `stm::stm` + `estimateEffect` +
  `findTopics`, `stmBrowser` (R); `stm` via rpy2, `bertopic`,
  `gensim` (Python).

## Assumptions & caveats

- **Covariate specification** — separate formulas for prevalence
  and content; misspecification biases both.
- **Number of topics `K`** — search via held-out likelihood /
  semantic coherence.
- **Interpretability** — inspect top words and representative
  documents for each topic before publication.
- **Content covariates** create separate topic-word distributions
  per group — needs more data.

## Related in this repo

- `topic-modeling-lda`, `topic-coherence-eval` — baseline LDA
  workflow.
- `word-embeddings`, `document-embedding-similarity` — neural
  alternatives.

## Run

```
python techniques/structural-topic-model/python/structural_topic_model.py
Rscript techniques/structural-topic-model/r/structural_topic_model.R
```

**Refs:** Roberts, M.E., Stewart, B.M., & Tingley, D. "stm: An R package for structural topic models." *Journal of Statistical Software*, 2019; Roberts, M.E., Stewart, B.M., Tingley, D. et al. "Structural topic models for open-ended survey responses." *American Journal of Political Science*, 2014.

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
