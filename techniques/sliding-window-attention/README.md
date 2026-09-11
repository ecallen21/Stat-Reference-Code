# Sliding-Window Attention (Reference §47.187)

Beltagy, Peters & Cohan (2020, Longformer); Jiang et al. (2023,
Mistral 7B). Restrict each token's attention to a fixed-size
**local window** of the past W tokens instead of the full history:

    attention_mask[i, j] = 1 if 0 ≤ i − j ≤ W else 0

Cost: O(T · W) instead of O(T²); memory / compute scale linearly in
sequence length. Combined with **attention sinks** (Xiao 2024),
enables streaming inference over very long contexts.

## Files

- `python/sliding_window_attention.py` — from-scratch causal
  sliding-window + optional sink tokens. T = 128, d = 32:
  - W = 8: rel-diff 1.37 (very lossy)
  - W = 32: rel-diff **0.54**
  - W = 64: rel-diff **0.24**
  - W = 16 + 4 sink tokens: **0.75 vs 0.91 without sinks**.
  - Compute at T = 8192: **16× cheaper** at W = 512, **64×
    cheaper** at W = 128.
- `r/sliding_window_attention.R` — no R port; recommends
  `transformers.Mistral`, `Longformer`, `FlashAttention`
  sliding-window kernels.

## When to use

- **Long-context LLMs** (Mistral 7B / Longformer / BigBird) where
  full-attention doesn't fit.
- **Streaming inference** — combined with sink tokens
  (StreamingLLM 2024) for effectively infinite context.
- **Efficient training** on very long sequences (documents, code
  files).

## When NOT to use

- **Tasks requiring long-range global dependencies** — sliding
  windows can't route information > W tokens away.
- **Short contexts** — full attention adds negligible cost.
- **Very lossy at small W** — quality degrades quickly for W < 32.

## Assumptions & caveats

- **Window size W** trades quality vs cost; typical 512-4096.
- **Attention sinks** (first 4 tokens) are essential for
  streaming — without them attention collapses.
- **Dilated / global tokens** (Longformer) mix local + a few
  global heads for long-range routing.
- **KV-cache still linear** — sliding attention doesn't help the
  cache size; combine with MQA / GQA.

## Related in this repo

- `flash-attention`, `attention-mechanism`,
  `transformer-encoder`, `transformer-decoder` — attention
  neighbours.
- `alibi-linear-attention-bias`, `rope-rotary-position-embedding`
  — position-encoding cousins for long-context.
- `paged-attention-vllm`, `multi-query-attention`,
  `kv-cache-quantization` — serving optimisations.

## Run

```
python techniques/sliding-window-attention/python/sliding_window_attention.py
Rscript techniques/sliding-window-attention/r/sliding_window_attention.R
```

**Refs:** Beltagy, I., Peters, M. E. & Cohan, A. "Longformer: The long-document transformer." *arXiv:2004.05150*, 2020; Jiang, A. Q. et al. "Mistral 7B." *arXiv:2310.06825*, 2023; Xiao, G., Tian, Y., Chen, B., Han, S. & Lewis, M. "Efficient streaming language models with attention sinks." *ICLR*, 2024.

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
