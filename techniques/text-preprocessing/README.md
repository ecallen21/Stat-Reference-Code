# Text Preprocessing (Reference §25.1)

Turn raw strings into normalised token lists for downstream models.

## Pipeline

| Step | Purpose |
|---|---|
| **Normalise** | Unicode NFKC, lowercase, strip punctuation. |
| **Tokenise** | split on whitespace (word), or use a subword tokeniser (BPE / WordPiece / SentencePiece). |
| **Stopword removal** | drop function words that carry little topical signal. |
| **Stem** | Porter / Snowball — rule-based suffix stripping. Cheap; produces non-words ("goes" → "goe"). |
| **Lemmatise** | dictionary + POS-tagged reduction to canonical form ("mice" → "mouse"). Slower; produces real words. |

## Classical vs modern NLP

- **Classical (bag-of-words) pipelines** — the pipeline above is standard. Stem or lemmatise depending on downstream needs (search / IR: stem; readability / NLU: lemma).
- **Modern (transformer) pipelines** — subword tokenisers preserve morphology, so stopword removal and stemming are usually skipped. Lowercasing depends on whether the model was pretrained cased or uncased.
- **Multilingual** — the Snowball family covers ~20 languages; SentencePiece works language-agnostically.

## When to use

- **TF-IDF / BM25 / topic-model / classical text classifier input** — always preprocess.
- **Transformer / neural encoder input** — use the model's own tokeniser; do not additionally stem or lemmatise.
- **Search engines** — stem + stopword-remove for the index; keep the raw text for display.
- **Analytics** — lemma → good for interpretable topics; stem → good for retrieval recall.

## Files

- `python/text_preprocessing.py` — from-scratch tokeniser + English stopword list + simplified Porter stemmer + small lemma map. Demo (3 English sentences): tokenises + strips stopwords + shows stem vs lemma side-by-side. Stemming reduces "goes"→"goe", "leaves"→"leave"; lemma map handles the classical irregulars (mice→mouse, geese→goose, children→child, was→be).
- `r/text_preprocessing.R` — `tokenizers::tokenize_words`, `tm::stopwords`, `SnowballC::wordStem`, `textstem::lemmatize_words`, `udpipe::udpipe_annotate`, `tidytext::unnest_tokens`, `quanteda::tokens`.

## Assumptions & caveats

- **Language-dependent** — this demo covers English. Other languages need appropriate stopword lists (`stopwords` package), stemmers (Snowball for many languages, `SudachiPy` / `Janome` for Japanese, jieba for Chinese), and lemmatisers (`stanza`, `udpipe`).
- **Aggressive normalisation loses signal** — email addresses, hashtags, URLs, numbers, and emoji often matter; strip only what you're sure about.
- **Casing matters** — "Apple" (company) vs "apple" (fruit); lowercasing all-caps text is fine, but consider preserving case for NER.
- **Contractions** — "don't" → ["don", "t"] under naive whitespace tokenisation; use a language-aware tokeniser (`nltk.word_tokenize`, `spacy`) if this matters.
- **Compound words** — Germanic languages need decompounding (`hunspell`, `charsplit`); ideographic languages need segmenters.
- **Stem vs lemma** — stemming is faster but produces non-words; lemma is slower but gives real words. Not interchangeable.

## Run

```
python techniques/text-preprocessing/python/text_preprocessing.py
Rscript techniques/text-preprocessing/r/text_preprocessing.R
```

**Refs:** Manning, C.D., Raghavan, P. & Schütze, H. *Introduction to Information Retrieval*, Cambridge UP, 2008; Porter, M.F. "An algorithm for suffix stripping." *Program* 14(3), 130–137, 1980; Bird, S., Klein, E. & Loper, E. *Natural Language Processing with Python*, O'Reilly, 2009.

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
