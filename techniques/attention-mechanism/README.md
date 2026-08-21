# Attention Mechanism (Reference §27.5)

Content-addressable "look up the relevant items and mix their values by
similarity to the query."

## Scaled dot-product attention

Given queries `Q ∈ ℝ^{n_q × d_k}`, keys `K ∈ ℝ^{n_k × d_k}`, values `V ∈ ℝ^{n_k × d_v}`:

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

- The `1 / √d_k` scaling stops dot products from saturating softmax at large `d_k`.
- Output has shape `(n_q, d_v)`.
- **Causal mask** (upper-triangular −∞) prevents position `i` from attending to positions > i — needed for autoregressive generation.

## Multi-head attention

Project `Q`, `K`, `V` into `h` heads with different linear maps; run scaled
dot-product attention in each head; concatenate and project back:

```
head_j = Attention(Q W_j^Q, K W_j^K, V W_j^V)
MHA(Q, K, V) = Concat(head_1, ..., head_h) · W^O
```

Different heads learn to attend to different relations (short-range vs
long-range, syntactic vs semantic).

## Self- vs cross-attention

- **Self-attention**: `Q = K = V` from the same source; used in encoder blocks and decoder self-attention.
- **Cross-attention**: `Q` from decoder, `K/V` from encoder; used in encoder-decoder transformers for translation.

## When to use

- **Sequence modelling** — the core of every modern language / vision transformer.
- **Set aggregation** — permutation-invariant pooling that respects content.
- **Retrieval and memory-augmented networks** — attention over a memory bank.
- **Multi-modal fusion** — cross-attention from one modality's queries to another's keys.

## Files

- `python/attention_mechanism.py` — from-scratch scaled dot-product attention + multi-head attention + causal mask. Demo:
  - Query = noisy copy of key 2, values are the standard basis → attention places 55.9% weight on position 2 as expected.
  - Multi-head self-attention (n=6, d_model=16, 4 heads) → row sums are 1.0.
  - Causal mask on n=6 → attention[3, :] zeros positions 4 and 5.
- `r/attention_mechanism.R` — `torch::nn_multihead_attention`, `keras3::layer_multi_head_attention`.

## Assumptions & caveats

- **Quadratic memory / compute in sequence length** — `n²` attention matrix; > 8k tokens gets expensive. Fixes: **Flash-Attention** (kernel-fused), **sparse attention** (BigBird, Longformer), **linear attention** (Performer, Linformer), **KV-cache** (autoregressive inference).
- **Softmax collapse** at large `d_k` without scaling — the `√d_k` factor is critical.
- **Positional encoding is separate** — attention itself is order-invariant; sinusoidal / learned / rotary positional embeddings inject order (`transformer-encoder`).
- **Attention weights ≠ explanations** — they show what the model *attended to*, not what caused the prediction (Jain-Wallace 2019).
- **Multi-head redundancy** — many heads are redundant; pruning them costs little accuracy.
- **Cross-attention needs shape compatibility** — `d_k` matches between Q and K; `d_v` can differ.

## Related in this repo

- `transformer-encoder` — a block built from multi-head self-attention + FFN + residual + LayerNorm.
- `embedding-layers` — the token-to-vector step that feeds attention.
- `centrality-measures` — PageRank as a "structural attention" analogue.

## Run

```
python techniques/attention-mechanism/python/attention_mechanism.py
Rscript techniques/attention-mechanism/r/attention_mechanism.R
```

**Refs:** Bahdanau, D., Cho, K. & Bengio, Y. "Neural machine translation by jointly learning to align and translate." *ICLR*, 2015; Vaswani, A. et al. "Attention is all you need." *NeurIPS*, 2017; Dao, T. et al. "FlashAttention: Fast and memory-efficient exact attention with IO-awareness." *NeurIPS*, 2022.

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
