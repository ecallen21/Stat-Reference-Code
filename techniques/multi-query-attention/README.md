# Multi-Query / Grouped-Query Attention (Reference §47.186)

Shazeer (2019, MQA); Ainslie et al. (2023, GQA). Multi-Head
Attention has H independent (Q, K, V) projections. **MQA** keeps H
query heads but **shares a single K, V** across all heads:

    MHA:   K, V shape (H, T, d_head)
    MQA:   K, V shape (   T, d_head)      -- 1/H the memory
    GQA:   K, V shape (G, T, d_head)      -- 1/(H/G) the memory

MQA has minor quality loss; **GQA** (G intermediate groups) recovers
most quality while keeping most of the memory / bandwidth win.

## Files

- `python/multi_query_attention.py` — from-scratch MHA / MQA / GQA
  attention. H = 32, T = 128, d_head = 64. KV-cache per layer at
  T = 2048:
  - MHA:   **33.6 MB**
  - GQA-8: **8.4 MB** (4× smaller)
  - GQA-4: **4.2 MB** (8× smaller)
  - MQA:   **1.0 MB** (32× smaller)
- `r/multi_query_attention.R` — no R port; recommends
  `transformers.LlamaAttention(num_key_value_heads=…)`,
  vllm / TensorRT-LLM kernels.

## When to use

- **Autoregressive LLM inference** — KV cache dominates memory
  at long context.
- **GQA** as the default trade — LLaMA-2 uses 8-way GQA
  (32 query heads, 8 KV groups).
- **When exact MHA quality doesn't matter** — most tasks.

## When NOT to use

- **Small models trained from scratch on small data** — MHA's
  extra capacity may help.
- **Cross-attention** (encoder-decoder) — MQA less common there.
- **When multi-modal / heterogeneous heads matter** — some heads
  intentionally specialise.

## Assumptions & caveats

- **Uptrain from MHA to GQA** — mean-pool KV weights of grouped
  heads, then fine-tune ~5 % of pretraining tokens (Ainslie 2023).
- **G choice** — 8 heads per group typical; smaller G
  = better quality, larger G = more memory saved.
- **Interaction with FlashAttention** — GQA / MQA reduce the K/V
  loads in the kernel too.

## Related in this repo

- `grouped-query-attention` — the H → G variant (same family
  covered in more detail).
- `flash-attention`, `attention-mechanism`,
  `transformer-encoder`, `transformer-decoder` — attention-family
  neighbours.
- `paged-attention-vllm`, `kv-cache-quantization` — KV-cache
  optimisation cousins.

## Run

```
python techniques/multi-query-attention/python/multi_query_attention.py
Rscript techniques/multi-query-attention/r/multi_query_attention.R
```

**Refs:** Shazeer, N. "Fast transformer decoding: One write-head is all you need." *arXiv:1911.02150*, 2019; Ainslie, J. et al. "GQA: Training generalized multi-query transformer models from multi-head checkpoints." *EMNLP*, 2023.

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
