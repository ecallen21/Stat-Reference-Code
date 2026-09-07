# Grouped-Query Attention (GQA) (Reference §47.37)

Ainslie et al. (2023, EMNLP). Compromise between multi-head
attention (each head has its own K, V) and multi-query attention
(all heads share one K, V):

    n_q_heads = 32
    n_kv_heads = 4        (each K, V shared by 8 Q-heads)

Cuts KV-cache memory by `n_q_heads / n_kv_heads` × with negligible
quality loss vs full MHA. Used in Llama 2 70B, Llama 3, Mistral,
Qwen, Gemma.

## Files

- `python/grouped_query_attention.py` — GQA forward pass from
  scratch. Demo (T=5, 8 Q-heads, 2 KV-heads, d=4): output shape
  matches MHA `(5, 32)`; KV-cache 80 bytes vs full MHA 320 → 4×
  smaller. Also runs MQA (n_kv_heads=1) → 8× smaller.
- `r/grouped_query_attention.R` — no R port; HF `transformers`
  LlamaAttention / MistralAttention, vLLM, flashattention
  (Python).

## When to use

- **Serving large LMs** — KV-cache dominates memory during
  autoregressive decoding.
- **Fine-tuning MHA models as GQA** — Ainslie 2023 method: mean-
  pool KV-heads within groups then continue training.
- **Any batch-1 latency-tight inference** where MHA cache is
  prohibitive.

## When NOT to use

- **Small models** where MHA cache is already tiny — no benefit.
- **Very short context** — KV-cache never dominates; MHA fine.
- **Quality-critical settings** with strict headroom — some tasks
  see slight regressions at extreme n_kv_heads ratios.

## Assumptions & caveats

- **`n_q_heads` divisible by `n_kv_heads`** — required for grouping.
- **RoPE ordering** — apply RoPE BEFORE the K-repeat / expansion
  step.
- **Quality-cache trade-off** — 1:8 (Llama 3 70B) is a common
  sweet spot; extreme MQA (1:32) can slightly hurt.
- **Training from scratch vs uptraining** — Ainslie shows you can
  convert MHA checkpoints with modest additional training.

## Related in this repo

- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder`, `rope-rotary-position-embedding`,
  `flash-attention` — transformer machinery.
- `speculative-decoding`, `inference-latency-profiling` — serving
  cousins.

## Run

```
python techniques/grouped-query-attention/python/grouped_query_attention.py
Rscript techniques/grouped-query-attention/r/grouped_query_attention.R
```

**Refs:** Ainslie, J. et al. "GQA: training generalized multi-query transformer models from multi-head checkpoints." *EMNLP*, 2023; Shazeer, N. "Fast transformer decoding: one write-head is all you need." *arXiv:1911.02150*, 2019.

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
