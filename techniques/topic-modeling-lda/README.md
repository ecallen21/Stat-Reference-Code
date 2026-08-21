# Latent Dirichlet Allocation — LDA (Reference §25.4)

Generative topic model (Blei-Ng-Jordan 2003) for a corpus of documents:

```
θ_d  ~ Dirichlet(α)                   document → topic proportions
φ_k  ~ Dirichlet(β)                    topic → word distributions
z_{d, i} ~ Categorical(θ_d)            topic of the i-th token in doc d
w_{d, i} ~ Categorical(φ_{z_{d, i}})    observed word
```

Each document is a mixture of a small number of topics; each topic is a
distribution over the vocabulary.

## Fitting

- **Collapsed Gibbs** (Griffiths-Steyvers 2004) — integrates `θ` and `φ` analytically; resamples `z_{d, i}` per token:

```
P(z_i = k | rest) ∝ (n_{d, k}^{−i} + α) · (n_{k, w}^{−i} + β) / (n_k^{−i} + V · β)
```

- **Variational EM** (Blei et al. 2003) — mean-field factorised approximate posterior; fast, deterministic.
- **Online / stochastic VI** (Hoffman-Blei-Bach 2010) — streaming corpora.
- **Structural topic model** (STM; Roberts et al. 2014) — LDA with covariates on topic prevalence / content.

## When to use

- **Exploratory analysis** of a document collection — recover latent themes.
- **Feature reduction** — replace TF-IDF's `|V|` dimensions with `K` topic proportions.
- **Metadata modelling** — topic prevalence over time, author, or region (`stm`).
- **Recommendation / soft clustering** of documents.
- **Pre-transformer default** — still competitive when interpretability > raw quality; **BERTopic** (transformer embeddings + HDBSCAN + class-based TF-IDF) is the modern successor.

## Files

- `python/topic_modeling_lda.py` — from-scratch collapsed-Gibbs LDA. Demo (D=80 documents, |V|=27, K=3 planted topics: sports / tech / food): recovers each true topic cleanly at the top of its `φ_k` — topic 0 = {score, match, season, team, goal, win}, topic 1 = tech, topic 2 = food. Matches sklearn's `LatentDirichletAllocation(learning_method='batch')` on all three topics (label order swapped, as expected).
- `r/topic_modeling_lda.R` — `topicmodels::LDA(method='Gibbs' | 'VEM')`, `lda::lda.collapsed.gibbs.sampler`, `stm::stm`.

## Assumptions & caveats

- **Bag of words** — order discarded; use bigram / trigram vocab if phrasing matters.
- **Choose K** — perplexity on held-out data + topic coherence + human interpretability (see `topic-coherence-eval`).
- **Priors matter** — small `α` (e.g. 0.1) → concentrated documents (few topics each); small `β` (e.g. 0.01) → concentrated topics (few words each).
- **Non-identifiability across runs** — topic labels are arbitrary; align by matching top-word overlap when comparing runs.
- **Long docs / short docs** — very short docs (tweets) have too little signal for standard LDA; use biterm models or short-text topic models.
- **Stopwords + rare words** — remove them first, they dominate raw topics.
- **BERTopic / NMF alternatives** — often produce cleaner topics on modern corpora, but LDA remains the textbook baseline.

## Related in this repo

- `text-preprocessing`, `tfidf-bm25` — the input pipeline.
- `document-clustering` — hard-clustering counterpart.
- `topic-coherence-eval` — evaluation of topic quality.
- `dirichlet-process-mixture` — non-parametric Bayesian alternative that infers K.
- `variational-inference` — the mean-field approximation used by the LDA VEM fitter.

## Run

```
python techniques/topic-modeling-lda/python/topic_modeling_lda.py
Rscript techniques/topic-modeling-lda/r/topic_modeling_lda.R
```

**Refs:** Blei, D.M., Ng, A.Y. & Jordan, M.I. "Latent Dirichlet allocation." *JMLR* 3, 993–1022, 2003; Griffiths, T.L. & Steyvers, M. "Finding scientific topics." *PNAS* 101(Suppl 1), 5228–5235, 2004; Roberts, M.E., Stewart, B.M. & Airoldi, E.M. "A model of text for experimentation in the social sciences." *JASA* 111(515), 988–1003, 2016.

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
