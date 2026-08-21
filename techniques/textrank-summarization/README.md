# TextRank Extractive Summarisation (Reference §25.12)

Mihalcea & Tarau (2004): pick the most **central** sentences from a document
by running PageRank on a sentence-similarity graph.

## Algorithm

1. Split document into sentences.
2. Vectorise each sentence (TF-IDF or bag-of-words).
3. Build a weighted undirected graph: nodes = sentences, edge weight `w_ij` = cosine similarity of sentences i and j (or an overlap ratio).
4. Run PageRank on the graph.
5. Return the top-K sentences (in original document order) as the summary.

## Related methods

- **LexRank** (Erkan & Radev 2004) — same graph-PageRank idea; usually with a threshold on similarity.
- **LSA-based summarisation** (Steinberger & Ježek) — SVD of the sentence-term matrix; sentences with highest projection on top singular components.
- **Extractive vs abstractive** — extractive returns exact sentences; abstractive (BART, T5, Pegasus, GPT) rewrites in new words.
- **Query-focused summarisation** — bias PageRank toward sentences relevant to a query (personalised PageRank).

## When to use

- **Long-document skimming** — press releases, meeting notes, scientific papers.
- **Cluster labelling** — top sentences of each cluster summarise the theme.
- **Cold-start** — extractive TextRank works with zero training data.
- **Multi-document summarisation** — concatenate + TextRank works surprisingly well.
- **Modern abstractive** — for polished summaries, fine-tune BART / T5 or call an LLM; extractive is faithful by construction (no hallucination) whereas abstractive can invent details.

## Files

- `python/textrank_summarization.py` — from-scratch sentence splitter + TF-IDF vectoriser + cosine-similarity graph + PageRank. Demo on an 8-sentence paragraph about renewable energy: selects the three most-central sentences (hydro / challenges / battery-technology) as the summary.
- `r/textrank_summarization.R` — `textrank::textrank_sentences`, `LexRankR::lexRank`, `LSAfun`; Python `sumy` and `summa` implementations; abstractive via `transformers.pipeline('summarization')` with BART / T5 / Pegasus.

## Assumptions & caveats

- **Sentence splitter matters** — abbreviations ("Dr.", "e.g.") and quotes trip naive `.!?` splitters; use `nltk.punkt` or `spacy.sents`.
- **Sentence similarity choice** — cosine on TF-IDF is standard; overlap-ratio (Mihalcea-Tarau original) works too; sentence embeddings (SBERT) give much better similarity.
- **Length normalisation** — long sentences win in raw overlap; divide by `log(|s|)` or use similarity that already normalises.
- **Extractive summaries are often disjointed** — no coherence guarantee between the chosen sentences.
- **Coverage / diversity** — greedy top-K can pick redundant sentences; use **MMR** (Maximal Marginal Relevance) to trade relevance vs novelty.
- **Domain adaptation** — the algorithm is unsupervised, so it transfers across domains and languages with just the tokeniser swapped out.

## Related in this repo

- `centrality-measures`, `hits-authority-hub` — graph-centrality building blocks.
- `tfidf-bm25` — the sentence vectoriser.
- `topic-modeling-lda` — a probabilistic alternative for "what's this document about?".
- `word-embeddings` — dense sentence embedding upgrade.

## Run

```
python techniques/textrank-summarization/python/textrank_summarization.py
Rscript techniques/textrank-summarization/r/textrank_summarization.R
```

**Refs:** Mihalcea, R. & Tarau, P. "TextRank: Bringing order into text." *EMNLP*, 2004; Erkan, G. & Radev, D.R. "LexRank: graph-based lexical centrality as salience in text summarization." *JAIR* 22, 457–479, 2004; Nenkova, A. & McKeown, K. *Automatic Summarization*, NOW Publishers, 2011.

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
