# Entity Linking — EL (Reference §25.x extra)

Given a **mention** in a context sentence, **link** it to a canonical entity
in a knowledge base (Wikidata, Wikipedia, UMLS, ChEBI, or a domain KB), or
return **NIL** if no match exists. Also called *wikification* or *named-entity
disambiguation* (NED).

## Two subtasks

1. **Candidate generation** — from the mention string, retrieve a shortlist of KB entities. Standard methods:
   - **Alias index** — precomputed lookup from surface forms to entity IDs.
   - **BM25 / TF-IDF** over entity names + aliases.
   - **Phonetic hashing** (Soundex, Metaphone).
   - **Neural bi-encoder** — embed mention + candidate; cosine top-k.

2. **Ranking / disambiguation** — pick the right entity from the shortlist using **context**:
   - **Bag-of-words** cosine against each entity's description (this demo).
   - **Cross-encoder** — concatenate (mention + context, candidate description); score jointly.
   - **Generative EL** (GENRE) — autoregressively decode the entity's Wikipedia title conditioned on context.

## Modern SOTA

- **BLINK** (Wu 2020) — Facebook's bi-encoder + cross-encoder pipeline over Wikipedia.
- **GENRE** (Cao 2021) — generative EL by decoding Wikipedia titles token-by-token.
- **ReFinED** (Ayoola 2022) — fast BERT-based end-to-end EL over Wikidata.

## When to use

- **Populate a knowledge graph** — EL is the standard pairing with NER.
- **Question answering / semantic search** — link mentions to KB entities so the retriever can hop through relations.
- **Coreference to a KB** — disambiguate "Apple" (company vs fruit) before further NLP.
- **Fact-checking / grounding** — cross-check claims by linking each entity to its KB entry.

## Files

- `python/entity_linking.py` — from-scratch **alias-index candidate generation** + **context-cosine ranker** over a mini 6-entity KB with three ambiguous surface forms (Michael Jordan basketball vs statistician; Apple company vs fruit; Paris France vs Texas). Demo: **6/6 correct** disambiguations across contexts.
- `r/entity_linking.R` — `reticulate` + `BLINK`, `GENRE`, `ReFinED`, `spaCy EntityLinker`; Wikidata / Wikipedia / DBpedia / YAGO / UMLS / ChEBI as knowledge bases.

## Assumptions & caveats

- **NIL handling** — mentions not in the KB should be labelled NIL, not force-matched.
- **Ambiguity is fundamentally context-dependent** — bag-of-words works for well-separated senses; harder cases (twin newspapers, obscure historical figures) need cross-encoders or LLMs.
- **KB coverage** — a mention absent from the KB will always get NIL or a wrong link; keep aliases and description text fresh.
- **Popularity prior** — many EL systems bias toward the more-linked candidate (`p(entity | mention) ∝ inbound-links(entity)`).
- **Cross-lingual EL** — link foreign-language mentions to English-Wikipedia titles (KB-Link, MEL, mGENRE).
- **Long-tail entities** — small businesses, minor towns, niche researchers underserved by BLINK / GENRE; domain-specific EL helps.

## Related in this repo

- `named-entity-recognition`, `coreference-resolution`, `relation-extraction` — the IE pipeline neighbours.
- `word-sense-disambiguation` — WSD is EL over word senses (WordNet) rather than KB entities.
- `sentence-similarity`, `word-embeddings`, `contrastive-learning` — the encoder families used to rank candidates.
- `question-answering` — RAG uses EL to filter retrieved passages by entity mention.

## Run

```
python techniques/entity-linking/python/entity_linking.py
Rscript techniques/entity-linking/r/entity_linking.R
```

**Refs:** Wu, L. et al. "Scalable zero-shot entity linking with dense entity retrieval (BLINK)." *EMNLP*, 2020; Cao, N.D. et al. "Autoregressive entity retrieval (GENRE)." *ICLR*, 2021; Ayoola, T. et al. "ReFinED: an efficient zero-shot-capable approach to end-to-end entity linking." *NAACL Industry*, 2022.

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
