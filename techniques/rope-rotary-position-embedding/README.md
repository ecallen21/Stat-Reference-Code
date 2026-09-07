# RoPE -- Rotary Position Embedding (Reference §47.36)

Su et al. (2021, *RoFormer*). Position information injected by
**rotating pairs** of embedding dimensions by an angle proportional
to position:

    q_pos = R(pos · θ) · q
    k_pos = R(pos · θ) · k
    θ_i = 10 000^{−2i / d}

Attention inner products then depend only on the **relative offset**
`(pos_q − pos_k)` — a natural translation-invariance bias.

Used in Llama, Mistral, Qwen, Gemma, GPT-NeoX, and most modern
open-source LLMs.

## Files

- `python/rope_rotary_position_embedding.py` — pair-rotation
  implementation from scratch. Demo (seq=6, d=8): shifts both Q
  and K positions by +2; the attention score matrix changes by
  `≤ 1.8e-15` — machine-precision relative-shift invariance.
- `r/rope_rotary_position_embedding.R` — no R port;
  `rotary-embedding-torch`, HF `transformers` Llama attention,
  xFormers, jax (Python).

## When to use

- **Autoregressive / bidirectional transformers** — standard in
  contemporary open LLMs.
- **Long-context extrapolation** — combined with NTK-aware or YaRN
  scaling extends beyond training length.
- **Whenever a translation-invariant positional bias is desired**.

## When NOT to use

- **Non-sequence data** — RoPE is 1-D by default; 2-D RoPE for
  images (Axial-RoPE) needed for vision transformers.
- **Absolute-position dependence required** — use learned
  positional embeddings.
- **Non-rotational geometry needs** — relative-attention bias
  (T5-style) may be simpler in some settings.

## Assumptions & caveats

- **d must be even** — pairs `(x[2i], x[2i+1])` rotated jointly.
- **Base 10 000** is default; smaller / larger bases trade high-
  frequency vs low-frequency positional signal.
- **Length extrapolation** — direct RoPE degrades beyond training
  length; use NTK-aware scaling, YaRN, or fine-tune with longer
  context.
- **Interaction with grouped-query attention** — apply RoPE
  BEFORE the K-repeat step in GQA.

## Related in this repo

- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder`, `grouped-query-attention`,
  `flash-attention` — transformer machinery.
- `mamba-state-space-transformer` — alternative long-context
  approach.
- `embedding-layers` — general embedding family.

## Run

```
python techniques/rope-rotary-position-embedding/python/rope_rotary_position_embedding.py
Rscript techniques/rope-rotary-position-embedding/r/rope_rotary_position_embedding.R
```

**Refs:** Su, J. et al. "RoFormer: enhanced transformer with rotary position embedding." *Neurocomputing*, 2024 (arXiv:2104.09864, 2021); YaRN: Peng, B. et al. "YaRN: efficient context window extension of large language models." *ICLR*, 2024.

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
