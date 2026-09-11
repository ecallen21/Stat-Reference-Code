# LLM.int8() / bitsandbytes (Reference §47.179)

Dettmers, Lewis, Belkada & Zettlemoyer (2022, NeurIPS). Two key
insights for lossless INT8 LLM inference:

1. **Mixed-precision decomposition** — identify OUTLIER FEATURES
   (a small set of columns with magnitudes ≫ 6) and keep them in
   FP16; quantise the rest to INT8.
2. **Vector-wise quantisation** — per-row of A, per-column of B.

Enables ~lossless (< 0.1 pt zero-shot degradation) INT8 inference
for models up to 175 B → half the memory of FP16.

## Files

- `python/bitsandbytes_int8_llm.py` — from-scratch matmul with
  outlier decomposition, 128 × 512 × 512 with 10 injected outlier
  input dims:
  - Naïve per-tensor INT8: **rel-err 0.051**.
  - LLM.int8() with outlier decomposition: **rel-err 0.0035** —
    ~15× lower error.
  - Memory saved: 51 % vs FP16 (outlier cols still FP16).
- `r/bitsandbytes_int8_llm.R` — no R port; recommends
  `bitsandbytes.nn.Linear8bitLt`, `transformers.load_in_8bit`.

## When to use

- **Serving LLMs on limited VRAM** — 175 B in 8-bit halves memory.
- **Fine-tuning via QLoRA** — 4-bit NF4 weights + BF16 LoRA
  adapters.
- **When perplexity must be preserved** — LLM.int8() is
  near-lossless vs naïve per-tensor INT8.

## When NOT to use

- **Very small models** — INT8 kernels don't beat FP16 tensor
  cores at small sizes.
- **Latency-critical low-batch** — dequant + kernel switch adds
  overhead.
- **When throughput matters more than memory** — GPTQ / AWQ are
  more inference-optimised.

## Assumptions & caveats

- **Outlier threshold** — paper uses 6.0; ~0.1-1 % of features are
  outliers at typical LLM sizes.
- **Vector-wise scales** need per-row / per-column dequant; extra
  storage vs per-tensor.
- **NF4 (4-bit normal float)** for QLoRA — a different quant
  scheme in the same library.
- **BitsandBytes** also provides 8-bit Adam / paged-Adam
  optimizers for memory-efficient training.

## Related in this repo

- `gptq-quantization`, `awq-quantization`,
  `product-quantization-pq`, `quantization-pruning` — weight-
  quantisation neighbours.
- `mixed-precision-training` — FP16 / BF16 training cousin.
- `lora-peft`, `prefix-prompt-tuning` — parameter-efficient
  fine-tuning cousins.

## Run

```
python techniques/bitsandbytes-int8-llm/python/bitsandbytes_int8_llm.py
Rscript techniques/bitsandbytes-int8-llm/r/bitsandbytes_int8_llm.R
```

**Refs:** Dettmers, T., Lewis, M., Belkada, Y. & Zettlemoyer, L. "LLM.int8(): 8-bit matrix multiplication for transformers at scale." *NeurIPS*, 2022; Dettmers, T., Pagnoni, A., Holtzman, A. & Zettlemoyer, L. "QLoRA: Efficient finetuning of quantized LLMs." *NeurIPS*, 2023.

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
