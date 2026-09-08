# word2vec Skip-Gram (SGNS) (Reference §47.117)

Mikolov, Chen, Corrado & Dean (2013a); Mikolov, Sutskever, Chen,
Corrado & Dean (2013b). Skip-gram + negative sampling:

    max_θ  E[log σ(v_c · v_w)] + k · E_{c' ~ P_noise}[log σ(−v_{c'} · v_w)]

where `v_w` is the INPUT (target) and `v_c` the OUTPUT (context)
embedding. Contrast with CBOW (context predicts word). Noise
distribution `P_noise ∝ p(w)^{0.75}`.

## Files

- `python/word2vec_skipgram.py` — from-scratch SGNS with SGD
  updates. Demo (3 semantic clusters × 5 words each, dim 16, 40
  epochs, window 3, k=3 negatives):
  - within-cluster cosine (king–queen 0.69, cat–dog 0.63, car–truck 0.67)
  - cross-cluster cosine (king–cat 0.31, cat–car 0.22, king–car 0.13).
- `r/word2vec_skipgram.R` — `text2vec::GlobalVectors` /
  `word2vec` (R); `gensim.models.Word2Vec`, `fasttext` (Python).

## When to use

- **Word / token embeddings** as pretrained features for NLP.
- **Item embeddings** in recommender systems (skip-gram over
  user-item sequences).
- **Graph embeddings** — node2vec / DeepWalk apply skip-gram to
  random walks.
- **Any co-occurrence-based representation learning**.

## When NOT to use

- **Contextual embeddings needed** — use BERT / RoBERTa; word2vec
  is one vector per word.
- **Very small corpora** — SGNS needs many context windows; use
  GloVe or subword models.
- **Downstream task with abundant supervised data** — end-to-end
  training may beat pretrained embeddings.

## Assumptions & caveats

- **Bag-of-context** — order within the window ignored.
- **Noise-distribution exponent** 0.75 empirically robust.
- **Dim choice** 100–300 typical; smaller for tiny vocabularies.
- **Subsampling frequent words** speeds training and often
  improves quality (Mikolov's threshold t = 1e-5).
- **Analogy structure** (king − man + woman ≈ queen) emerges when
  training data is large and diverse.

## Related in this repo

- `word-embeddings`, `sentence-similarity`,
  `tfidf-bm25`, `text-preprocessing`,
  `text-preprocessing-pipeline`, `byte-pair-encoding-bpe`,
  `masked-language-modeling`, `transformer-encoder` — NLP
  representation stack.
- `node2vec-deepwalk`, `graph-neural-network`,
  `latent-space-network` — graph-embedding cousins that reuse
  skip-gram.
- `random-fourier-features`, `random-projections`,
  `product-quantization-pq` — embedding-compression neighbours.

## Run

```
python techniques/word2vec-skipgram/python/word2vec_skipgram.py
Rscript techniques/word2vec-skipgram/r/word2vec_skipgram.R
```

**Refs:** Mikolov, T., Chen, K., Corrado, G. & Dean, J. "Efficient estimation of word representations in vector space." *ICLR (Workshop)*, 2013; Mikolov, T., Sutskever, I., Chen, K., Corrado, G. & Dean, J. "Distributed representations of words and phrases and their compositionality." *NeurIPS*, 2013.

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
