# Entity Embedding Layers (Reference §27.12)

Turn a categorical variable with `K` levels into a **learnable dense vector**
of dimension `d ≪ K`. The embedding layer is a `K × d` lookup matrix trained
jointly with the downstream head.

## Why not one-hot?

- **Parameter count** — one-hot × dense linear layer of width `h` costs `K · h` parameters; embedding + linear costs `K · d + d · h`, cheaper when `K >> d`.
- **Similarity learned** — levels that behave similarly end up close in embedding space; one-hot treats every level as equidistant.
- **Transferability** — pretrained embeddings (word2vec, BERT tokens, product2vec) transfer across tasks; one-hot representations don't.
- **Interpretability** — plot embeddings (t-SNE / UMAP) to inspect learned structure.

## Rule of thumb

- `d ≈ min(50, ⌈K^0.25⌉)` — Guo-Berkhahn (2016) heuristic.
- `d = 300` for words (GloVe, word2vec).
- `d = 768–4096` for transformer tokens.
- L2-regularise the embedding matrix (weight decay) to avoid over-fitting rare levels.

## Common uses

- **Text tokens** (word / subword embeddings) — see `word-embeddings`.
- **Recommendation** — user × item factorisation is exactly two embedding matrices.
- **Tabular data** — high-cardinality categoricals (zip code, product ID, hospital ID). Alternative: target encoding, hashing.
- **Graphs** — node embeddings (see `graph-embedding-spectral`).

## Files

- `python/embedding_layers.py` — from-scratch entity-embedding layer + softmax head + scatter-add backprop. Demo: 12 categories arranged in 3 latent groups (levels 0–3, 4–7, 8–11 map to classes 0, 1, 2); after 400 epochs at `dim=2`, the model reaches 100% train and test accuracy and the learned embeddings cleanly cluster into 3 regions (within-class cosine +0.999, across-class −0.498).
- `r/embedding_layers.R` — `torch::nn_embedding`, `keras3::layer_embedding`; Python `torch.nn.Embedding`, `tensorflow.keras.layers.Embedding`, `fastai TabularModel`.

## Assumptions & caveats

- **Cold-start** for unseen categories — no vector exists; fall back to a shared `<UNK>` embedding or hash the category into a fixed bucket.
- **Rare-level over-fit** — a category with 2 training examples gets a highly noisy vector; L2 weight decay or minimum-frequency thresholding helps.
- **Scale mismatch with other features** — embeddings often have larger magnitudes than standardised numeric features; concatenate carefully or use LayerNorm after.
- **Dimension choice** — too small under-fits interactions; too large adds parameters without gain. Search over 4, 8, 16, 32.
- **Frozen vs fine-tuned** — pretrained embeddings often benefit from starting frozen and unfreezing later in training.
- **Interpretability** — cluster / project embeddings after training to inspect learned structure.

## Related in this repo

- `word-embeddings` — text-token embedding via word2vec.
- `graph-embedding-spectral` — node embeddings for graphs.
- `deep-mlp-backprop`, `transformer-encoder` — architectures where embedding layers appear as the first block.
- `text-classification`, `named-entity-recognition` — downstream tasks that consume embeddings.

## Run

```
python techniques/embedding-layers/python/embedding_layers.py
Rscript techniques/embedding-layers/r/embedding_layers.R
```

**Refs:** Bengio, Y. et al. "A neural probabilistic language model." *JMLR* 3, 1137–1155, 2003; Guo, C. & Berkhahn, F. "Entity embeddings of categorical variables." *arXiv:1604.06737*, 2016; Cheng, H.-T. et al. "Wide & deep learning for recommender systems." *DLRS*, 2016.

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
