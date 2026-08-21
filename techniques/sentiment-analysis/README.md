# Sentiment Analysis (Reference §25.7)

Estimate the affective polarity of a text — positive, negative, or neutral —
plus (optionally) intensity or emotion category.

## Two approaches

### Lexicon-based (VADER-style)

```
polarity(doc) = Σ_w lexicon[w] · (negation_flip if any) · intensifier_multiplier
```

- Small human-curated dictionary of scored words (`good` = +1.8, `terrible` = −2.8).
- **Valence shifters**: negations flip sign (`not bad` → positive), intensifiers scale magnitude (`very` = 1.3).
- Fast, transparent, needs no labels; poor on sarcasm and domain-specific vocabulary.

### Supervised (LR / SVM / transformer)

- Train on labelled data with any text classifier (see `text-classification`).
- Beats lexicons when domain-labelled data exists; requires labels; opaque coefficients (unless L1 sparse).

### Modern default

Fine-tune a pretrained transformer (e.g. `cardiffnlp/twitter-roberta-base-sentiment-latest`, `siebert/sentiment-roberta-large-english`) or use it zero-shot via a pipeline. Gets ~90–95% F1 on standard benchmarks vs 65–80% for lexicons.

## When to use

- **Cold start / no labels** — start with a lexicon.
- **Domain-specific text** (medical, legal, financial) — always fine-tune a supervised model; general-purpose lexicons under-perform.
- **Real-time / high-throughput** — lexicon + LR-on-TF-IDF for CPU-only serving.
- **State-of-the-art** — transformer pipeline; batched inference on GPU.

## Files

- `python/sentiment_analysis.py` — from-scratch lexicon scorer (with negation + intensifier handling) + supervised LR baseline via TF-IDF. Demo: correctly flags "This movie was absolutely amazing…" as +7.42, "Terrible product, would not recommend…" as −2.8, "Not bad at all…" as +2.0 (negation flip). Supervised LR hits 100% on the 20-doc test set of the disjoint positive / negative vocabulary demo.
- `r/sentiment_analysis.R` — `tidytext::get_sentiments`, `sentimentr::sentiment`, `syuzhet::get_sentiment`, `vader::get_vader`; Python `nltk.sentiment.vader`, `TextBlob`, `transformers.pipeline('sentiment-analysis')`.

## Assumptions & caveats

- **Sarcasm and irony** are still hard even for large transformers.
- **Domain shift** — a movie-review-trained model transfers poorly to tweets or clinical notes.
- **Aspect-based sentiment** (product × attribute) needs a two-stage extractor + polarity classifier, not a document-level score.
- **Multilingual sentiment** — lexicons exist for many languages (`syuzhet`, `sentimentr` covers a few); transformers cover more via multilingual models (XLM-R).
- **Emoji / emoticons** — VADER handles them explicitly; lexicons that don't miss the strongest signal on social media.
- **Neutral is often mis-modelled** — many training corpora only label pos/neg; the neutral class is often a leftover with no consistent semantics.

## Related in this repo

- `text-preprocessing`, `tfidf-bm25` — the input pipeline.
- `text-classification`, `naive-bayes`, `logistic-regression` — supervised alternatives.
- `calibration-scaling` — get well-calibrated pos/neg probabilities.

## Run

```
python techniques/sentiment-analysis/python/sentiment_analysis.py
Rscript techniques/sentiment-analysis/r/sentiment_analysis.R
```

**Refs:** Pang, B. & Lee, L. "Opinion mining and sentiment analysis." *Foundations and Trends in Information Retrieval* 2(1-2), 1–135, 2008; Hutto, C.J. & Gilbert, E. "VADER: a parsimonious rule-based model for sentiment analysis of social media text." *ICWSM*, 2014; Liu, B. *Sentiment Analysis: Mining Opinions, Sentiments, and Emotions*, 2nd ed., Cambridge UP, 2020.

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
