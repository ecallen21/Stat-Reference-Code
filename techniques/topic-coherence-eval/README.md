# Topic-Coherence Evaluation (Reference §25.11)

Automatic metrics for the interpretability of a topic model's topics. Each
topic is represented by its **top-N words**; a coherence score summarises how
strongly they co-occur.

## Common metrics

### UMass (Mimno et al. 2011)

```
c_UMass(W) = (2 / N(N − 1)) · Σ_{i<j} log( (D(w_i, w_j) + 1) / D(w_j) )
```

- `D(w)` = # documents containing `w`; `D(w_i, w_j)` = # docs with both.
- Uses only the **training corpus** — no external reference.
- Higher (closer to 0) = more coherent.

### UCI / PMI (Newman et al. 2010)

```
c_UCI(W) = (2 / N(N − 1)) · Σ_{i<j} PMI(w_i, w_j)
PMI(a, b) = log(p(a, b) / (p(a) p(b)))
```

- Uses a **large external reference corpus** (Wikipedia is standard).
- Positive = words co-occur more than chance.

### c_v (Röder et al. 2015)

Combines sliding-window co-occurrence + NPMI + cosine similarity of topic vectors. Correlates highest with human judgement in the Röder benchmark; the recommended default when a large reference corpus is available.

### Perplexity

Standard held-out log-likelihood metric, `exp(−log_lik / n_tokens)`. Lower is
better in the model's own eyes, but **Chang et al. 2009** famously showed that
lower perplexity can accompany *less* human-interpretable topics — so
coherence has replaced perplexity as the primary evaluation.

## When to use

- **Choose K** — sweep topic counts; pick the K near the coherence peak.
- **Compare hyperparameters** — `α`, `β`, or LDA vs NMF vs BERTopic.
- **Filter bad topics** — drop topics with UMass < some cutoff.
- **Human validation** — pair with **word intrusion** or **topic intrusion** tasks; automatic metrics are a proxy for interpretability, not a replacement.

## Files

- `python/topic_coherence_eval.py` — UMass + UCI-PMI coherence + perplexity. Demo (200 docs from 3 disjoint topical vocabularies): three true topics score UMass ≈ −0.10, UCI-PMI ≈ +1.1 (excellent); a mixed-vocab bad topic scores UMass = −3.3, UCI-PMI = −21.9 — coherence clearly separates good from bad.
- `r/topic_coherence_eval.R` — `textmineR::CalcProbCoherence`, `topicdoc::topic_coherence`, `gensim.models.CoherenceModel` with `coherence in {'u_mass', 'c_v', 'c_uci', 'c_npmi'}`.

## Assumptions & caveats

- **Corpus dependence** — UMass is corpus-specific; UCI needs a large external reference for stable estimates.
- **N choice** matters — top-5, top-10, top-20 give different rankings; pick and stick.
- **Repetition and near-duplicates** in a topic** inflate coherence artificially — dedupe top words first.
- **Semantic vs statistical coherence** — automatic metrics catch co-occurrence, not concept coherence; human eval remains the arbiter.
- **c_v vs UMass** — c_v tracks human judgement better but needs external data and is more expensive to compute.
- **Perplexity ≠ interpretability** (Chang et al. 2009). Don't optimise K on perplexity alone.

## Related in this repo

- `topic-modeling-lda` — the model this evaluates.
- `dirichlet-process-mixture` — non-parametric alternative that chooses K automatically.
- `variational-inference` — the LDA training loop whose objective is the ELBO (a related but distinct quantity).

## Run

```
python techniques/topic-coherence-eval/python/topic_coherence_eval.py
Rscript techniques/topic-coherence-eval/r/topic_coherence_eval.R
```

**Refs:** Mimno, D. et al. "Optimizing semantic coherence in topic models." *EMNLP*, 2011; Newman, D. et al. "Automatic evaluation of topic coherence." *NAACL*, 2010; Röder, M., Both, A. & Hinneburg, A. "Exploring the space of topic coherence measures." *WSDM*, 2015; Chang, J. et al. "Reading tea leaves: How humans interpret topic models." *NeurIPS*, 2009.

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
