# Transformer Encoder Block (Reference §27.6)

The building block of BERT, RoBERTa, T5-encoder, ViT, and every modern
"encoder-style" transformer. Composed of:

```
x' = LayerNorm(x + MultiHeadSelfAttention(x))
y  = LayerNorm(x' + FeedForward(x'))
where FeedForward(z) = Linear₂(GELU(Linear₁(z)))
```

Two conventions:

- **Post-norm** (Vaswani 2017 original): `x + Sublayer(LN(x))` … `LN(x + Sublayer(x))`. Common in early BERT / GPT-2.
- **Pre-norm** (Xiong 2020, used here): `LN` *before* each sublayer. Trains more stably at depth, tolerates larger learning rates without warmup collapse. Modern default (GPT-2 large, LLaMA, ViT-Base).

## Positional encoding

Attention alone is permutation-invariant; add position information:

- **Sinusoidal** (Vaswani 2017): `PE[pos, 2i] = sin(pos / 10000^(2i/d))`, `cos` in the odd channels.
- **Learned** absolute embeddings (BERT).
- **Rotary (RoPE)** — rotate query/key vectors by position-dependent angle; encodes *relative* position and scales to unseen lengths.
- **ALiBi** — additive linear bias in attention scores; extrapolates well.

## When to use

- **Encoder-only** — classification, sequence tagging, embedding models (BERT, RoBERTa, DeBERTa, ModernBERT, sentence-transformers).
- **Encoder-decoder** — seq2seq translation, summarisation (T5, BART, mT5).
- **Decoder-only (causal)** — language modelling, generative chat (GPT, LLaMA, Claude, Mistral). Same block, causal mask, cross-attention removed.
- **Vision / audio / multi-modal** — ViT patch tokens, wav2vec, CLIP, DALL-E, LLaVA.

## Files

- `python/transformer_encoder.py` — from-scratch pre-norm encoder block: multi-head self-attention + position-wise FFN (GELU) + residuals + LayerNorm + sinusoidal positional encoding. Demo (n=8, d_model=32, 4 heads, d_ff=64): input row-variance 0.52 → output row-variance 1.0000 (LayerNorm working); PE[0, :4] = [0, 1, 0, 1] sin/cos alternation; 4 stacked blocks reproduce shape (8, 32) with row-variance 1.
- `r/transformer_encoder.R` — `torch::nn_transformer_encoder_layer / nn_transformer_encoder`, `keras3::layer_multi_head_attention + layer_normalization`.

## Assumptions & caveats

- **Quadratic in sequence length** — see `attention-mechanism` for linearised / sparse variants.
- **LayerNorm placement** — pre-norm has become the de-facto default; the two versions have measurably different loss landscapes.
- **FFN inner dim** is usually `4 · d_model`; sometimes 2× to save FLOPs.
- **Weight initialisation** — small std (Xavier / Kaiming with fan_in) matters more than in CNNs; too-large init collapses attention.
- **Warmup schedule** — linear warmup for first ~1–5% of steps, then cosine or inverse-sqrt decay; skipping warmup often diverges post-norm models.
- **Positional-encoding choice** interacts with length generalisation — RoPE and ALiBi extrapolate; sinusoidal doesn't beyond training length; learned absolute doesn't at all.
- **Decoder blocks** additionally include causal self-attention and encoder-cross-attention sublayers.

## Related in this repo

- `attention-mechanism` — the multi-head attention primitive.
- `embedding-layers` — the token / position embeddings that feed the encoder.
- `deep-mlp-backprop`, `dropout-batchnorm`, `adam-optimizer` — the training add-ons.
- `text-classification`, `topic-modeling-lda`, `sentiment-analysis` — downstream NLP tasks a transformer encoder feeds.

## Run

```
python techniques/transformer-encoder/python/transformer_encoder.py
Rscript techniques/transformer-encoder/r/transformer_encoder.R
```

**Refs:** Vaswani, A. et al. "Attention is all you need." *NeurIPS*, 2017; Devlin, J. et al. "BERT: Pre-training of deep bidirectional transformers for language understanding." *NAACL*, 2019; Xiong, R. et al. "On layer normalization in the transformer architecture." *ICML*, 2020.

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
