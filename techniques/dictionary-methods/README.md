# Dictionary-Based Text Scoring (Reference §42.14)

Pennebaker et al. (2015), Young & Soroka (2012). Count words
belonging to predefined semantic categories and report the
proportion per document — the **LIWC** paradigm. Fast, transparent,
and reproducible; quality depends entirely on **dictionary
coverage** and **domain match**.

## Metric

```
score(document, category) = |tokens ∈ category| / |tokens|
```

## When to use

- **High-transparency reporting** — every classification decision
  is traceable to a word list.
- **Historical / longitudinal corpora** where consistent scoring
  matters more than accuracy.
- **Small domains** with a validated dictionary (mental health,
  political affect).

## When NOT to use

- **Context-heavy meaning** — "not happy" scores positive; use
  transformer sentiment analysis.
- **Domain mismatch** — general-purpose LIWC on legal / biomedical
  text under-covers domain vocabulary.
- **Sarcasm, negation, complex sentiment** — dictionaries are
  blind to these.

## Files

- `python/dictionary_methods.py` — 5-category LIWC-style lexicon
  (positive/negative emotion, health, certainty, hedging) + per-
  document proportions. Demo (4 clinical / everyday sentences):
  positive emotion 0.20 on "great improvement, love"; health 0.17
  on the pain / hospital sentence; certainty 0.33 on "definitely
  certain"; hedging 0.27 on "perhaps ... might ... possibly".
- `r/dictionary_methods.R` — `quanteda::dictionary` +
  `tokens_lookup`, `tidytext::get_sentiments`, LIWC commercial
  (R); `empath`, LIWC via API, `nltk` custom dictionaries (Python).

## Assumptions & caveats

- **Coverage** — always report the proportion of tokens matching
  any dictionary category so readers know the effective sample.
- **Negation blindness** — combine with a NegEx-style scan or use
  ML for context-sensitive tasks.
- **Word senses** — polysemy inflates false-positive matches
  ("depression" as illness vs terrain).
- **Validation** — cross-check dictionary output against a hand-
  coded subset before publication.

## Related in this repo

- `sentiment-analysis` — dictionary is a sentiment special case.
- `keyness-analysis` — statistical comparison of corpora.
- `text-preprocessing-pipeline` — tokenisation upstream.

## Run

```
python techniques/dictionary-methods/python/dictionary_methods.py
Rscript techniques/dictionary-methods/r/dictionary_methods.R
```

**Refs:** Pennebaker, J.W., Boyd, R.L., Jordan, K., & Blackburn, K. *The Development and Psychometric Properties of LIWC2015*, University of Texas at Austin, 2015; Young, L. & Soroka, S. "Affective news: the automated coding of sentiment in political texts." *Political Communication*, 2012.

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
