# Semantic Textual Similarity (Reference §25.x extra)

Score how similar two texts are in meaning, not surface form.

## Bag-of-embeddings baseline

```
sent_vec(s) = mean_{w ∈ s} vec(w)       (or IDF-weighted, or SIF)
sim(a, b)   = cosine( sent_vec(a), sent_vec(b) )
```

Works well for short texts (< 30 tokens) — often within 5–10 points of
learned sentence-transformer embeddings on STS benchmarks.

## SIF (Arora et al. 2017)

Improvement on the mean:

1. Weight each word by `a / (a + p(w))` (down-weights common words).
2. Compute the weighted mean.
3. Subtract the projection on the first principal component (removes the
   dominant "topic" direction).

## Sentence-BERT and successors

Fine-tune BERT with a **triplet or contrastive** loss so `cos(sent_A, sent_B)`
correlates with human similarity — the "sentence-transformer" family
(SBERT, SimCSE, MPNet, GTE, E5, INSTRUCTOR, BGE, mxbai, nomic-embed,
voyage-3, text-embedding-3). Current SOTA on STS-B ~ 89 Pearson.

## Cross-encoder vs bi-encoder

- **Bi-encoder** — encode each text separately; cheap; used for retrieval.
- **Cross-encoder** — concatenate both texts and score jointly; expensive; used for reranking.
- **ColBERT / late-interaction** — encode separately, combine at query time.

## When to use

- **Semantic search** — retrieve documents by similarity to a query.
- **Deduplication** — find near-duplicate texts (support tickets, news articles).
- **Retrieval-augmented generation (RAG)** — embed corpus + query, cosine-rank top-k.
- **Semantic clustering** — cluster documents by embedding-space distance.
- **Paraphrase detection**, **entailment**, **STS regression**.

## Files

- `python/sentence_similarity.py` — from-scratch bag-of-embeddings + cosine. Toy 8-D embeddings where semantically similar words are placed near each other in embedding space. Demo pairs: within-topic (animal / tech / food) pairs score 0.996–0.999; cross-topic pairs score ~0.53 — the baseline correctly separates domains.
- `r/sentence_similarity.R` — `text::textEmbed`, `text2vec::sim2`; Python `sentence-transformers`, `openai` / `voyage` / `anthropic` embedding APIs.

## Assumptions & caveats

- **Static embeddings miss polysemy** — "bank" (river) vs "bank" (finance) get the same vector; contextual embeddings (BERT, SBERT) resolve this.
- **Length bias** — longer texts drift toward the corpus mean; SIF and PCA whitening help.
- **Anisotropy** — embedding vectors concentrate in a narrow cone; whitening or standardisation before cosine improves scores.
- **Cross-lingual STS** requires multilingual embeddings (LaBSE, multilingual-E5, XLM-R).
- **Domain mismatch** — a Wikipedia-trained embedder underperforms on clinical / legal / code text; fine-tune on domain pairs.
- **Bench dependence** — STS-B / SICK / MTEB give different rankings; report multiple.

## Related in this repo

- `text-preprocessing`, `word-embeddings` — inputs.
- `tfidf-bm25` — sparse baseline for retrieval.
- `document-clustering`, `text-classification` — downstream tasks that consume similarity or embeddings.
- `contrastive-learning` — the training recipe behind modern sentence encoders.

## Run

```
python techniques/sentence-similarity/python/sentence_similarity.py
Rscript techniques/sentence-similarity/r/sentence_similarity.R
```

**Refs:** Arora, S., Liang, Y. & Ma, T. "A simple but tough-to-beat baseline for sentence embeddings." *ICLR*, 2017; Reimers, N. & Gurevych, I. "Sentence-BERT: Sentence embeddings using Siamese BERT-networks." *EMNLP*, 2019; Wang, L. et al. "Text embeddings by weakly-supervised contrastive pre-training." *arXiv:2212.03533* (E5), 2022.

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
