# KV-Cache Quantization (Reference §47.174)

Sheng et al. (2023, FlexGen); Liu et al. (2023, KIVI). LLM inference
is memory-bound; the K, V cache grows as O(L · T · H · d).
Quantising the cache to INT8 / INT4 halves (or quarters) memory
and bandwidth while preserving generation quality:

    K, V (FP16) → per-token / per-channel affine INT8:
    q = round((x − z) / s),   x̂ = q · s + z

Recovery quality depends on granularity: **per-channel** for K
(preserves attention-score scale) and **per-group / per-token** for
V (KIVI 2023).

## Files

- `python/kv_cache_quantization.py` — from-scratch INT8 per-channel
  quant for K + INT4 per-group quant for V, T = 256 tokens, d = 64.
  Attention-output relative error vs FP32:
  - INT8 K only: **0.0017** (essentially exact).
  - INT4 V only: **0.0661**.
  - INT8 K + INT4 V: **0.0661** (V dominates).
  - Memory: 56 % saved with both quantised.
- `r/kv_cache_quantization.R` — no R port; recommends `vllm`
  (built-in KV modes), `FlexGen`, `llama.cpp` Q4_0.

## When to use

- **Long-context inference** — 32k / 128k contexts blow up KV
  memory; quantisation makes them fit.
- **Batch LLM serving** — more concurrent sessions in the same
  VRAM budget.
- **Consumer / edge deployment** — 7-13 B models with INT4
  KV cache run on 8-16 GB.

## When NOT to use

- **Very short contexts** — KV isn't the bottleneck.
- **When perfect logit reproducibility is required** —
  quantisation adds small stochasticity.
- **Prompt evaluation with tiny batch size** — compute-bound,
  not memory-bound.

## Assumptions & caveats

- **Per-channel K** is standard; per-token K loses too much
  granularity for softmax accuracy.
- **INT4 is enough for V** (paper) but K needs INT8 or per-group
  INT4.
- **Group size 32-128** on V; larger = smaller scale footprint but
  higher error.
- **Dequantise on the fly** during attention; some kernels keep
  compute in INT-space via marlin / vllm awq kernels.

## Related in this repo

- `paged-attention-vllm` — sibling memory-management technique.
- `bitsandbytes-int8-llm`, `gptq-quantization`,
  `awq-quantization`, `quantization-pruning` — weight-side
  quantisation cousins.
- `flash-attention` — combines with kernel-level quant KV.

## Run

```
python techniques/kv-cache-quantization/python/kv_cache_quantization.py
Rscript techniques/kv-cache-quantization/r/kv_cache_quantization.R
```

**Refs:** Sheng, Y. et al. "FlexGen: High-throughput generative inference of large language models with a single GPU." *ICML*, 2023; Liu, Z. et al. "KIVI: A tuning-free asymmetric 2-bit quantization for KV cache." *ICML*, 2024.

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
