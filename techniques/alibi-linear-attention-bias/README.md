# ALiBi — Attention with Linear Biases (Reference §47.170)

Press, Smith & Lewis (2022). Replaces learned / sinusoidal position
embeddings with a fixed **linear bias** on attention scores:

    A_ij = softmax(Q_i K_jᵀ / √d − m_h · |i − j|)

where m_h = 2^(−8 h / H) for head h (geometrically spaced slopes).
No trainable positional parameters. **Extrapolates** to sequences
longer than seen at training — unlike learned / RoPE embeddings.

## Files

- `python/alibi_linear_attention_bias.py` — from-scratch multi-
  head ALiBi self-attention with causal masking. 4-head slopes =
  [0.25, 0.0625, 0.0156, 0.0039]. Per-head mean attention radius:
  - T = 32:  [2.8, 6.0, 7.3, 7.6]
  - T = 128: [3.3, 12.3, 24.8, 30.0]
  - **T = 512**: [3.5, 14.7, 50.4, **100.3**]
  - Head 0 stays local (~3 tokens), head 3 attends far (~100
    tokens at T = 512) — the slopes create a natural short-to-long
    frequency ladder.
- `r/alibi_linear_attention_bias.R` — pure-R slope computation +
  library recs (`transformers` Bloom / MPT / OPT).

## When to use

- **Length-extrapolation** required — a model trained on T = 2048
  must serve T = 8192.
- **Causal decoder-only LLMs** (Bloom, MPT, OPT variants) — ALiBi
  is a default there.
- **When absolute position matters less than relative** (most
  language tasks).

## When NOT to use

- **Bidirectional encoders** — ALiBi biases both directions
  symmetrically; less commonly used.
- **Tasks needing precise absolute position** (some form-parsing,
  code, chess).
- **Very short contexts** — no extrapolation gain to be had.

## Assumptions & caveats

- **Slopes m_h = 2^(-8 h / H)** are the paper's default; not
  learned.
- **No positional embedding at all** — ALiBi is added directly to
  the attention scores.
- **Extrapolation quality decays** past ~4-8× the training
  length; not unlimited.
- **BF16 storage** — biases are added in FP32 to preserve
  precision.

## Related in this repo

- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder`, `flash-attention`,
  `grouped-query-attention`, `mamba-state-space-transformer` —
  Transformer neighbours.
- `rope-rotary-position-embedding` — rotary position embedding
  cousin (extrapolates less naturally, better at fine positional
  arithmetic).
- `rmsnorm-normalization` — LLaMA-family sibling normalisation.

## Run

```
python techniques/alibi-linear-attention-bias/python/alibi_linear_attention_bias.py
Rscript techniques/alibi-linear-attention-bias/r/alibi_linear_attention_bias.R
```

**Refs:** Press, O., Smith, N. A. & Lewis, M. "Train short, test long: Attention with linear biases enables input length extrapolation." *ICLR*, 2022; Su, J. et al. "RoFormer: Enhanced transformer with rotary position embedding." *Neurocomputing* 568, 2024.

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
