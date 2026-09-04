# Collocation Statistics — PMI & Dunning G² (Reference §42.12)

Dunning (1993), Manning & Schütze (1999 ch 5). Identify
statistically-significant **word pairs** (bigrams, skip-grams,
n-grams) in a corpus.

## Metrics

- **Pointwise mutual information** (PMI):
  `PMI(x, y) = log(P(x, y) / (P(x) · P(y)))`. Simple but unstable
  for rare pairs.
- **Dunning log-likelihood ratio** (`G²`): 2 × 2 contingency table
  test comparing observed and expected co-occurrence; robust at
  low counts.
- **Chi², t-score**: alternative test statistics on the same table.

## When to use

- **Phrase detection** — extend TF-IDF or LDA with bigrams like
  "chest pain" as single features.
- **Terminology extraction** — identify domain-specific multi-word
  units.
- **Corpus linguistics** — quantify collocation strength beyond
  eye-balling.

## When NOT to use

- **Modern subword tokenisers** (BPE / WordPiece) already learn
  phrase-like units.
- **Very small corpora** where every bigram is unique.

## Files

- `python/collocation_pmi.py` — PMI + Dunning G² over unigram /
  bigram counts (custom). Demo (8 short clinical / everyday
  sentences): bigrams "left arm" (n=2), "radiating to", "numbness
  noted", "arm numbness" top the PMI / G² ranks respectively.
- `r/collocation_pmi.R` — `quanteda::textstat_collocations`,
  `tidytext::unnest_tokens` (R); `nltk.collocations`,
  `gensim.Phrases`, `sklearn.CountVectorizer(ngram_range)`
  (Python).

## Assumptions & caveats

- **Independence** under H₀ — collocations violate it; that's the
  point.
- **Low-count sensitivity** — PMI is unstable at rare pair counts
  (n ≤ 5); prefer Dunning G² or apply a minimum-count filter.
- **Stop words** dominate raw bigram counts; either filter them
  out or use content-word patterns.
- **Multiple testing** — correct across the vocabulary of
  candidate bigrams (BH).

## Related in this repo

- `ngram-language-model`, `keyness-analysis`, `kwic-concordance`
  — cousin corpus tools.
- `word-embeddings`, `document-embedding-similarity` — modern
  neural alternatives.

## Run

```
python techniques/collocation-pmi/python/collocation_pmi.py
Rscript techniques/collocation-pmi/r/collocation_pmi.R
```

**Refs:** Dunning, T. "Accurate methods for the statistics of surprise and coincidence." *Computational Linguistics*, 1993; Manning, C.D. & Schütze, H. *Foundations of Statistical Natural Language Processing*, MIT Press, 1999 (ch 5).

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
