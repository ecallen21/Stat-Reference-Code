# Word-Sense Disambiguation — WSD (Reference §25.x extra)

Choose the intended sense of an ambiguous word given its context. Classical
example: "bank" → financial-institution vs river-bank.

## Two classical approaches

### Lesk (1986)

Score each candidate sense by the count of overlapping words between its
**dictionary gloss** and the target word's **context**:

```
best_sense = argmax_s | context(w) ∩ gloss(s) |
```

Extended Lesk (Banerjee-Pedersen 2003) adds glosses of related senses
(hypernyms, hyponyms, meronyms) to enrich the overlap.

### Embedding-based

Average the pretrained embeddings of the context tokens; average the
embeddings of each sense's gloss tokens; pick the sense with the highest
cosine:

```
ctx_vec  = mean_{w ∈ ctx} vec(w)
gloss_vec_s = mean_{w ∈ gloss(s)} vec(w)
best_sense = argmax_s cos( ctx_vec, gloss_vec_s )
```

Strong pre-transformer baseline; still competitive when a well-tuned
embedding table is available.

## Modern SOTA

- **GlossBERT** (Huang 2019) — BERT sentence-pair over (context, gloss) fine-tuned for binary sense-match.
- **BEM** (Blevins-Zettlemoyer 2020) — bi-encoder mapping context and gloss into a shared space.
- **ConSeC** (Barba 2021) — current SOTA on the ALL-WSD benchmark.
- **Zero-shot LLM prompting** — supply the sense definitions; ask the model to classify. Often near-SOTA without any WSD training.

## When to use

- **Downstream MT / IR** — disambiguate ambiguous terms before translation / retrieval.
- **Preprocessing for knowledge-graph population** — link mentions to senses / entities.
- **Lexicography and corpus linguistics** — study sense distributions.
- **Modern practice**: use a fine-tuned or zero-shot LLM for anything past a demo.

## Files

- `python/word_sense_disambiguation.py` — from-scratch Lesk (gloss-overlap) + bag-of-embeddings cosine WSD on a toy "bank" dataset with financial vs river senses. Both approaches get 4/4 test cases correct; overlaps and cosines shown per prediction.
- `r/word_sense_disambiguation.R` — `wordnet`, `reticulate` + `nltk.wsd.lesk / pywsd / transformers`; modern references (GlossBERT, BEM, ConSeC).

## Assumptions & caveats

- **Lesk is fragile** on real data — vocabulary mismatch between context and gloss; short glosses give near-random overlap.
- **Sense inventories matter** — WordNet's fine-grained senses often disagree with human intuition; coarser inventories (OntoNotes coarse) reach higher agreement.
- **Data sparsity** — most senses of most words appear rarely in labelled corpora; the most-frequent-sense baseline is very strong on rare senses.
- **Cross-lingual WSD** — align senses via BabelNet / ConceptNet / bilingual dictionaries.
- **Word-in-context (WiC)** — a related task: given two sentences, do they use the target word in the same sense?
- **Static embeddings mix senses** — a single vector for "bank" averages over senses; contextual embeddings (BERT, RoBERTa) provide per-instance sense vectors.

## Related in this repo

- `text-preprocessing`, `word-embeddings`, `sentence-similarity` — the pipeline.
- `named-entity-recognition`, `pos-tagging` — companion sequence-labelling tasks.
- `masked-language-modeling`, `transformer-encoder` — modern representation source for WSD.

## Run

```
python techniques/word-sense-disambiguation/python/word_sense_disambiguation.py
Rscript techniques/word-sense-disambiguation/r/word_sense_disambiguation.R
```

**Refs:** Lesk, M. "Automatic sense disambiguation using machine readable dictionaries." *SIGDOC*, 1986; Banerjee, S. & Pedersen, T. "Extended gloss overlaps as a measure of semantic relatedness." *IJCAI*, 2003; Huang, L. et al. "GlossBERT: BERT for word sense disambiguation with gloss knowledge." *EMNLP*, 2019.

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
