# Word Embeddings — Word2Vec / GloVe / FastText (Reference §25.3)

Dense low-dimensional vector representations of words such that semantically
similar words have similar vectors.

## Skip-Gram with Negative Sampling (SGNS)

Mikolov et al. (2013b). For each observed (word `w`, context `c`) pair from a
sliding window, and `k` random "negatives" `c'` sampled from `p(w)^0.75`:

```
max Σ_{(w, c) ∈ D+}  log σ(v_w · u_c)
    + Σ_{(w, c') ∈ D−} log σ(−v_w · u_{c'})
```

Fitted by mini-batch SGD; `v_w` is the "center" embedding, `u_c` the "context"
embedding. Usually only `v_w` is used downstream.

## GloVe (Pennington-Socher-Manning 2014)

Fit `v_w · u_c ≈ log X_{wc}` where `X_{wc}` is the global co-occurrence count.
A weighted least-squares view; captures ratios of co-occurrence.

## FastText (Bojanowski et al. 2017)

Word2Vec on **character n-grams**: word vector = sum of its subword vectors.
Handles OOV words and rare morphology.

## Contextual embeddings

BERT / RoBERTa / DeBERTa / GPT-family / sentence-transformers. Each occurrence
of a word gets its own vector; huge improvement on downstream NLU but
requires large pretrained models.

## When to use

- **Static embeddings** (Word2Vec / GloVe / FastText) — cheap, interpretable, good for classical retrieval + classification + analogy.
- **Contextual embeddings** — needed for anything semantics-heavy (QA, NLI, sense disambiguation).
- **Hybrid** — TF-IDF for sparse lexical + dense sentence-transformers for semantic; standard in modern IR (BM25 + bi-encoder rerankers).

## Files

- `python/word_embeddings.py` — from-scratch SGNS with negative sampling, trained by SGD. Demo (toy corpus with two disjoint domains: animals + tech, ~300 sentences each): after training, `cat` → rabbit / hops / swims (animal cluster) at top; `server` → crashes / api / cloud (tech cluster) at top; cross-domain words appear only lower. Cross-check with gensim `Word2Vec(sg=1, negative=5)` matches the top neighbours when available.
- `r/word_embeddings.R` — `text2vec::GloVe`, `wordVectors::train_word2vec`, `fastTextR::ft_train`, `textdata::embedding_glove6b`.

## Assumptions & caveats

- **Corpus size drives quality** — Word2Vec / GloVe want ~100M+ tokens for stable analogies. The demo corpus here is a proof-of-concept, not a serious embedding.
- **Distributional hypothesis** — "you know a word by the company it keeps." Fine for topical similarity, weaker for syntactic role and antonymy (embeddings often place antonyms *close* rather than far).
- **Anisotropy** — trained embeddings live in a narrow cone; cosine similarities are always positive and clustered. Whitening (Su et al. 2021) or normalisation helps.
- **Bias** — word embeddings inherit societal biases from training corpora; debiasing methods (Bolukbasi 2016) exist but are imperfect.
- **Static vs contextual** — a static Word2Vec `bank` vector conflates river-bank and financial-bank. Use contextual encoders for polysemy.
- **Hyperparameters matter more than the algorithm** — window, dim, negatives, min-count, subsampling threshold; grid-search on downstream task.

## Related in this repo

- `text-preprocessing`, `tfidf-bm25` — sparse alternatives.
- `document-clustering`, `text-classification`, `topic-modeling-lda` — downstream tasks that consume embeddings.
- `graph-embedding-spectral` — spectral analogue for graphs (nodes-in-context ≈ words-in-context).

## Run

```
python techniques/word-embeddings/python/word_embeddings.py
Rscript techniques/word-embeddings/r/word_embeddings.R
```

**Refs:** Mikolov, T. et al. "Efficient estimation of word representations in vector space." *ICLR Workshop*, 2013a; Mikolov, T. et al. "Distributed representations of words and phrases and their compositionality." *NeurIPS*, 2013b; Pennington, J., Socher, R. & Manning, C.D. "GloVe: Global vectors for word representation." *EMNLP*, 2014; Bojanowski, P. et al. "Enriching word vectors with subword information." *TACL* 5, 135–146, 2017.

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
