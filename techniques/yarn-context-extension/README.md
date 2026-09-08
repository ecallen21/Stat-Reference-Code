# YaRN Context Extension (Reference §47.49)

Peng, Quesnelle, Sharkey & Chan (2023). Extends the effective
context window of a RoPE-based LLM (originally L, e.g. 2048)
by up to 16× WITHOUT catastrophic perplexity blow-up. Three
ingredients:

1. **NTK-aware interpolation** — preserve high-freq RoPE bands,
   scale down low freqs (interpolation).
2. **NTK-by-parts** — smoothly ramp the interpolation on per-
   dimension wavelength vs original context length.
3. **Attention-temperature scaling** — `t = 0.1 · log(s) + 1` to
   preserve softmax entropy at the extended context.

Empirically preserves in-context recall much better than pure
positional interpolation (PI) or linear scaling.

## Files

- `python/yarn_context_extension.py` — YaRN frequency ramp +
  temperature computation from scratch. Demo (dim=64, orig
  ctx=2048):
  - scale 1× → 100 % freqs preserved, temp 1.000
  - scale 16× → 34 % high freqs preserved, temp 1.277, max
    wavelength grows linearly with s (753 876 tokens).
- `r/yarn_context_extension.R` — no established R
  implementation; Python `transformers`, `vllm`.

## When to use

- **Long-context fine-tuning** — extend a 2 k model to 8 k / 32 k /
  128 k.
- **Retrieval-augmented pipelines** — need to stuff many docs into
  the prompt.
- **Function-calling / tool-use** where prompts run long.

## When NOT to use

- **Short-context tasks** — no benefit; use base RoPE.
- **Custom position encodings** (ALiBi, T5-relative) — YaRN is
  RoPE-specific.
- **Untrained scale factors** — YaRN needs a short fine-tune
  (~400 steps in the paper) to fully stabilise.

## Assumptions & caveats

- **α, β ramp parameters** default to (1, 32); paper heuristic —
  tune per model if aliasing artefacts appear.
- **Attention temperature** — increases with `log(scale)`; needed
  to preserve entropy or the model over-focuses on nearest tokens.
- **Fine-tune step** typically required for best perplexity;
  training-free YaRN degrades on tasks needing long-range retrieval.
- **KV cache size** grows linearly with context — VRAM budget still
  binds.

## Related in this repo

- `rope-rotary-position-embedding` — the base positional
  encoding.
- `attention-mechanism`, `grouped-query-attention`,
  `flash-attention` — Transformer efficiency stack.
- `mixture-of-experts`, `speculative-decoding`,
  `mamba-state-space-transformer` — long-context / scaling
  alternatives.
- `transformer-encoder`, `transformer-decoder`, `bert`, `gpt` —
  base architectures.

## Run

```
python techniques/yarn-context-extension/python/yarn_context_extension.py
Rscript techniques/yarn-context-extension/r/yarn_context_extension.R
```

**Refs:** Peng, B., Quesnelle, J., Sharkey, H. & Chan, S. "YaRN: Efficient context window extension of large language models." *arXiv:2309.00071*, 2023; Chen, S. et al. "Extending context window of large language models via positional interpolation." *arXiv:2306.15595*, 2023.

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
