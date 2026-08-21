# Coreference Resolution (Reference §25.x extra)

Link mentions in a discourse that refer to the same real-world entity:

- **"Alice went to the store. She bought milk."** → `{Alice, She}`
- **"Bob loves his dog. The dog barks."** → `{Bob, his}`, `{dog, The dog}`

## Classical mention-pair approach

For each ordered pair `(candidate_antecedent, anaphor)` train a **binary
classifier** on features:

- **Gender / number / animacy agreement**.
- **String / head match**.
- **Distance** between mentions.
- **Anaphor-is-pronoun** flag.
- **Syntactic-role match** (subject / object) — from POS + parse.

At inference: for each new mention `m`, score all previous mentions and link
to the highest-scoring compatible antecedent (or open a new cluster if all
scores fall below a threshold). Cluster mentions transitively.

## Modern models

- **End-to-end coreference** (Lee 2017) — span embeddings from BiLSTM, joint scoring of spans and pair-links.
- **s2e-coref** (Kirstain 2021) — start-to-end span scoring with efficient factorisation.
- **LingMess** (Otmazgin 2023) — multi-expert scoring per mention pair; SOTA on OntoNotes.
- **Zero-shot LLM prompting** — feed the text and ask for cluster IDs. Competitive on simple documents.

## When to use

- **Information extraction** — populate a knowledge graph with entity mentions.
- **Question answering** — resolve pronouns before matching against a knowledge base.
- **Summarisation** — merge mentions to compute per-entity attention.
- **Machine translation** — pronoun resolution across languages (English "it" ↔ French "il / elle").
- **Redaction / de-identification** — remove all mentions of an entity, not just the first.

## Files

- `python/coreference_resolution.py` — from-scratch rule-based mention-pair scorer with gender / number / animacy / head-match / distance features + greedy transitive clustering. Demo on 3 toy documents:
  - "Alice / Alice / she / her" → one cluster (all four).
  - "Bob / his dog / The dog / Bob" → `{Bob, Bob}` + `{his dog, The dog}` (correct).
  - "Carol / David / she / him" → `{Carol, she}` + `{David, him}` (correct gender resolution).
- `r/coreference_resolution.R` — no strong native R package; use `reticulate` + `spacy` + `coreferee`, `allennlp`, `fastcoref`, `s2e-coref`, or `LingMess`.

## Assumptions & caveats

- **Mention detection is upstream** — this module takes mentions as input; real systems detect them from NER + noun-phrase parsing + pronoun lists.
- **Gender / number agreement is language-specific** and often ambiguous ("they" as singular).
- **Named entities vs pronouns vs definite noun phrases** all need different features; the mention-pair model needs to distinguish them.
- **Long-distance coreference** — pronouns can refer arbitrarily far back; distance features help but heavy-tailed reference distances are hard.
- **Bridging anaphora** ("the door" referring to a previously-mentioned house) is harder than direct coreference; usually handled separately.
- **Zero anaphora** (dropped subject in Japanese, Chinese, Italian) needs language-specific tooling.
- **Evaluation** — CoNLL-2012 MUC / B³ / CEAF metrics; report the average (CoNLL F1).

## Related in this repo

- `pos-tagging`, `named-entity-recognition`, `syntactic-parsing-cky` — pipeline upstream of coref.
- `text-preprocessing`, `word-embeddings`, `sentence-similarity` — features / representations.
- `hmm` — sequence-model neighbour; the entity-mention model (Ng-Cardie) is HMM-adjacent.

## Run

```
python techniques/coreference-resolution/python/coreference_resolution.py
Rscript techniques/coreference-resolution/r/coreference_resolution.R
```

**Refs:** Hobbs, J.R. "Resolving pronoun references." *Lingua* 44, 311–338, 1978; Soon, W.M., Ng, H.T. & Lim, D.C.Y. "A machine learning approach to coreference resolution of noun phrases." *Computational Linguistics* 27(4), 521–544, 2001; Lee, K. et al. "End-to-end neural coreference resolution." *EMNLP*, 2017.

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
