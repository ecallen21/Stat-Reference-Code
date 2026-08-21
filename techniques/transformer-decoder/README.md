# Transformer Decoder Block (Reference §27.x extra)

Vaswani-style decoder block, pre-norm variant. The **causal** counterpart of
`transformer-encoder`. Powers GPT, LLaMA, Mistral, DeepSeek, Qwen, and
every modern chat / completion model.

## Structure

```
x' = LayerNorm(x + MaskedMultiHeadSelfAttention(x))       ← causal mask
x' = LayerNorm(x' + MultiHeadCrossAttention(x', memory))  ← encoder-decoder only
y  = LayerNorm(x' + FeedForward(x'))
```

- **Masked self-attention** — upper-triangular mask prevents position `i` from attending to positions `> i`. Essential for autoregressive generation.
- **Cross-attention** — queries from decoder, keys / values from encoder memory. Used in encoder-decoder models (T5, BART, Whisper) and removed in decoder-only models (GPT / LLaMA).
- **Feed-forward** — position-wise MLP; usually `4 · d_model` wide with GELU or SwiGLU activation.

## Encoder-decoder vs decoder-only

| Family | Cross-attention | Example |
|---|---|---|
| Encoder-decoder | present | T5, BART, mT5, Whisper, NLLB |
| Decoder-only (causal LM) | **removed** | GPT-2/3/4, LLaMA, Mistral, Qwen, DeepSeek, Claude, Gemini |

## Standard extensions

- **RoPE** — rotary positional encoding (used in LLaMA / Qwen / Mistral).
- **ALiBi** — additive linear bias on attention scores.
- **GQA / MQA** — grouped-query / multi-query attention; cheaper KV memory.
- **Flash-Attention** — kernel-fused attention, memory-efficient.
- **KV-cache** — reuse the K and V matrices across decoding steps; essential for real-time inference.
- **SwiGLU / GeGLU** — gated activation in the FFN; slightly better than plain GELU.

## Files

- `python/transformer_decoder.py` — from-scratch pre-norm decoder block with masked self-attention + cross-attention + FFN; runs both encoder-decoder and decoder-only variants on the same input. Demo:
  - Shapes preserved through the block.
  - LayerNorm output row-variance = 1 as expected.
  - **Causality verified**: perturbing `x[4]` non-uniformly leaves positions 0–3 exactly unchanged (masked out); positions 4–5 change.
- `r/transformer_decoder.R` — `torch::nn_transformer_decoder_layer / nn_transformer_decoder`; Python `torch.nn.TransformerDecoderLayer`, `huggingface transformers.GPT2Block / LlamaDecoderLayer`.

## Assumptions & caveats

- **Causal mask correctness is easy to break** — verify with the perturbation test above; a single off-by-one turns your LM into a leakage nightmare.
- **KV-cache correctness** — bugs in cache reuse produce subtle wrong-answer regressions; test by running with cache off vs on and checking token-level agreement.
- **Positional encoding compatibility** — rotary / ALiBi / sinusoidal are not interchangeable at inference; must match training.
- **Pre-norm vs post-norm** — modern decoders are almost universally pre-norm for stability at depth.
- **Cross-attention shape** — the memory (K/V) may have different length from the query; different heads still valid.
- **Decoder-only with prefix / prompt** is standard for chat; keep the prompt in the KV cache and only decode new tokens.

## Related in this repo

- `attention-mechanism`, `transformer-encoder` — the sibling architecture and its underlying attention.
- `text-generation-decoding` — greedy / beam / top-k / nucleus / speculative decoding on top of a decoder LM.
- `masked-language-modeling` — the bidirectional counterpart training objective.
- `lr-schedules`, `adam-optimizer`, `residual-connections`, `dropout-batchnorm` — training-loop pairings.

## Run

```
python techniques/transformer-decoder/python/transformer_decoder.py
Rscript techniques/transformer-decoder/r/transformer_decoder.R
```

**Refs:** Vaswani, A. et al. "Attention is all you need." *NeurIPS*, 2017; Radford, A. et al. "Language models are unsupervised multitask learners (GPT-2)." OpenAI TR, 2019; Touvron, H. et al. "LLaMA: open and efficient foundation language models." *arXiv:2302.13971*, 2023.

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
