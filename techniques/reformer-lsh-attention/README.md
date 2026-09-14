# Reformer LSH Attention (Reference §47.384)

Kitaev, Kaiser & Levskaya (2020 ICLR). Approximate softmax
attention in `O(n log n)` by hashing queries and keys into
buckets with LOCALITY-SENSITIVE HASHING (LSH):

```
signs = (Q · R) > 0     # random projections R
bucket(q) = signed-bit-string → integer bucket
attend only within same bucket
```

Combined with REVERSIBLE RESIDUAL LAYERS (activation reuse to
avoid storing intermediates), Reformer trains 64 k-token
sequences on a single GPU.

## Files

- `python/reformer_lsh_attention.py` — 256 tokens, `d=32`,
  tied Q=K (Reformer's setup). Relative output error is
  ≤ 1.5 % across 4/8/16 buckets while using only 6.8 %
  of the score pairs (4 500 / 65 536) — the LSH bucketing
  preserves the dominant softmax mass.
- `r/reformer_lsh_attention.R` — no first-class R package;
  reticulate to `transformers.ReformerModel` (R);
  `transformers.ReformerModel`, `x_transformers`, from-scratch
  (Python).

## When to use

- **Very long sequences** — 8 k+ tokens where full attention
  is `O(n²)` prohibitive.
- **Tied Q = K** — Reformer's assumption; simplifies bucketing.
- **When approximate attention is acceptable** — most large
  language / genomics models tolerate small error.
- **Combined with reversible layers** — enables single-GPU
  training on very long context.

## When NOT to use

- **Short sequences (≤ 1024)** — vanilla attention is faster.
- **When attention entropy is high** (uniform) — LSH loses
  mass; use Longformer / Performer.
- **Non-symmetric Q ≠ K** — need Reformer's shared-weights
  design.

## Assumptions & caveats

- **Number of hashing rounds** — 4-8 rounds standard; more
  rounds ⇒ more coverage of near-neighbours.
- **Bucket size / count** — target bucket size around
  `n / (n_buckets)` = 128 for `n = 8k, n_buckets = 64`.
- **Softmax entropy** — LSH depends on attention being
  peaked; if it is diffuse (early training), fallback to
  full attention.
- **Reversible layers** — the memory saving requires them;
  vanilla non-reversible layers must store activations.
- **Causal masking** — sort by bucket, then apply causal mask
  chunk-by-chunk.

## Related in this repo

- `longformer-sparse-attention`, `linformer-projection`,
  `performer-random-features`, `nystromformer-approximation` —
  efficient-attention siblings.
- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder`, `flash-attention` — attention
  neighbours.
- `min-hash-lsh` — LSH-family cousin outside NLP.
- `hnsw-ann-search`, `product-quantization-pq` — related
  ANN indexes.

## Run

```
python techniques/reformer-lsh-attention/python/reformer_lsh_attention.py
Rscript techniques/reformer-lsh-attention/r/reformer_lsh_attention.R
```

**Refs:** Kitaev, N., Kaiser, Ł. and Levskaya, A. "Reformer: the efficient transformer." In *ICLR*, 2020.

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
