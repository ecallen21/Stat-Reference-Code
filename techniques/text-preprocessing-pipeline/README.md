# Text Preprocessing Pipeline (Reference §42.11)

Manning-Raghavan-Schütze (2008 ch 2), Welbers-van Atteveldt-Benoit
(2017). The standard NLP preprocessing chain:

1. **Tokenise** — split text into tokens (words, subwords,
   sentences).
2. **Normalise** — lower-case, strip punctuation.
3. **Stop-word removal** — drop function words.
4. **Stem** (Porter / Snowball) — crude morphological reduction
   ("running" → "run", "bett" for "better").
5. **Lemmatise** (WordNet, spaCy) — dictionary-based reduction to
   the lemma ("better" → "good", "children" → "child") requiring
   POS + vocabulary.

## When to use

- **TF-IDF / bag-of-words** — stemming reduces vocabulary size.
- **Retrieval / classification** — some normalisation almost always
  helps.

## When NOT to use

- **Transformer models** — modern BERT / GPT tokenisers handle
  subwords; do NOT lower-case or stem before feeding a pretrained
  model.
- **Domain preservation** — dropping digits kills dose extraction;
  stemming destroys drug names; stop-word removal deletes negation
  ("no"). Preprocessing is task-dependent.

## Files

- `python/text_preprocessing_pipeline.py` — tokeniser, stop-word
  filter, compact Porter-suffix stemmer, tiny lemma dictionary.
  Demo on a mixed sentence shows stem ("running → runn") is
  aggressive-and-lossy; lemma ("running → run", "better → good")
  is semantically clean.
- `r/text_preprocessing_pipeline.R` — `quanteda::tokens` +
  `tokens_wordstem`, `tidytext::unnest_tokens`, `tm::tm_map`,
  `SnowballC` (R); `nltk` (PorterStemmer, SnowballStemmer,
  WordNetLemmatizer), `spacy` (Python).

## Assumptions & caveats

- **Order matters** — lower-case then punctuation strip, else
  contractions break.
- **Stopword lists differ** — English NLTK ≠ spaCy ≠ quanteda; pin
  one for reproducibility.
- **Stemming vs lemmatising** — stems are non-words; lemmas are
  dictionary words. Report which you used.
- **Never do this before a transformer** — modern models rely on
  original casing and punctuation.

## Related in this repo

- `tfidf-bm25`, `topic-modeling-lda`, `named-entity-recognition`
  — downstream users.
- `word-embeddings` — modern alternative; usually skip preprocessing.

## Run

```
python techniques/text-preprocessing-pipeline/python/text_preprocessing_pipeline.py
Rscript techniques/text-preprocessing-pipeline/r/text_preprocessing_pipeline.R
```

**Refs:** Manning, C.D., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval*, Cambridge University Press, 2008 (ch 2); Welbers, K., van Atteveldt, W., & Benoit, K. "Text analysis in R." *Communication Methods and Measures*, 2017.

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
