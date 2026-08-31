# Relation Extraction — RE (Reference §25.x extra)

Given a sentence and **two entity mentions**, classify the relation type
between them from a fixed inventory (or return "no relation").

## Classical approaches

### Pattern / rule-based (Hearst 1992)

Hand-written surface patterns like:

- `X, [the] founder of Y` → `founder_of(X, Y)`
- `X was born in Y` → `born_in(X, Y)`
- `X acquired Y` → `acquired(X, Y)`

Fast to prototype, interpretable, high precision on covered patterns, low recall on out-of-pattern phrasings.

### Distant supervision (Mintz 2009)

Align a knowledge-base tuple `(X, r, Y)` to sentences that mention both `X` and `Y`; treat every such sentence as a **positive** training example for `r`. Noisy but scalable.

### Supervised neural

- **PCNN** (Zeng 2015) — piecewise CNN over the sentence.
- **BERT-EM / EM+MTB** (Soares 2019) — inject entity markers, fine-tune a transformer classifier.
- **REBEL** (Cabot 2021) — seq2seq: generate the entire relation triple set.
- **Joint NER + RE** (JEREX, PL-Marker, W2NER) — end-to-end.

## When to use

- **Populate a knowledge graph** — RE is the standard IE downstream of NER.
- **Fact extraction from text** — biomedical entity relations, business acquisitions, legal contracts.
- **Question answering pipelines** — RE + entity linking = structured knowledge for lookup.
- **NOT** as the sole source for high-stakes decisions — RE has 60–80% F1 on realistic benchmarks; always human-review.

## Files

- `python/relation_extraction.py` — from-scratch rule-based RE with 5 regex patterns (`founder_of`, `ceo_of`, `born_in`, `acquired`, `headquartered_in`). Demo on 6 sentences:
  - `Bill Gates, the founder of Microsoft, stepped down in 2000` → `founder_of(Bill Gates, Microsoft)` ✓
  - `Marie Curie was born in Warsaw` → `born_in(Marie Curie, Warsaw)` ✓
  - `Google acquired YouTube in 2006 for $1.65 billion` → `acquired(Google, YouTube)` ✓
  - `Elon Musk, the CEO of Tesla, spoke at the conference` → `ceo_of(Elon Musk, Tesla)` ✓
  - `Amazon is headquartered in Seattle` → `headquartered_in(Amazon, Seattle)` ✓
  - `The Beatles were a British rock band` → (no known relation matched) ✓
- `r/relation_extraction.R` — `reticulate` + `OpenNRE`, `spaCy REL`, `DeepKE`, `huggingface transformers` for TACRED / SemEval-2010.

## Assumptions & caveats

- **Rule-based** is high-precision, low-recall — always add a fallback classifier for messy real text.
- **Distant supervision** injects label noise; multi-instance learning (Riedel 2010) mitigates.
- **Long-distance relations** need cross-sentence / document-level RE (DocRED benchmark).
- **N-ary relations** (drug-target-disease) need graph-of-events models.
- **Zero-shot RE** with an LLM prompt is a reasonable modern default; validate on your relation inventory.
- **Entity boundaries** are upstream; RE errors compound with NER errors.

## Related in this repo

- `named-entity-recognition` — supplies the mentions RE classifies.
- `coreference-resolution` — link "he" to "Elon Musk" before RE.
- `entity-linking` — attach mentions to a knowledge-base entity.
- `question-answering`, `text-classification` — downstream users of RE.

## Run

```
python techniques/relation-extraction/python/relation_extraction.py
Rscript techniques/relation-extraction/r/relation_extraction.R
```

**Refs:** Hearst, M.A. "Automatic acquisition of hyponyms from large text corpora." *COLING*, 1992; Mintz, M. et al. "Distant supervision for relation extraction without labeled data." *ACL-IJCNLP*, 2009; Baldini Soares, L. et al. "Matching the blanks: distributional similarity for relation learning." *ACL*, 2019; Cabot, P.-L. & Navigli, R. "REBEL: relation extraction by end-to-end language generation." *EMNLP Findings*, 2021.

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
