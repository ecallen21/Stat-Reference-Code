# RMSNorm — Root Mean Square Layer Normalisation (Reference §47.171)

Zhang & Sennrich (2019). Simplifies LayerNorm by dropping the mean-
centring:

    LayerNorm:  y = (x − μ) / σ · γ + β
    RMSNorm:    y = x / RMS(x) · γ     where RMS(x) = √(mean(x²))

Same computational form but **7-64 % faster** and equivalent quality
on Transformers. Used in LLaMA, GPT-NeoX, T5-v1.1, Mistral, Gemma.

## Files

- `python/rmsnorm_normalization.py` — from-scratch LayerNorm and
  RMSNorm, 1000 forward passes on a (128, 512) tensor:
  - LayerNorm output: mean 0.000, std 1.000 (centres and scales).
  - RMSNorm  output: mean 0.245, std 0.970 (only scales).
  - Timing: LayerNorm **392 ms** vs **RMSNorm 170 ms** — **56.6 %
    faster** for this workload.
  - Norm ‖y‖ is identical (22.63) across input scales — both
    normalise magnitude, but RMSNorm preserves mean.
- `r/rmsnorm_normalization.R` — pure-R LayerNorm vs RMSNorm.

## When to use

- **Transformer LLMs and MoE variants** where the mean-subtract is
  redundant and expensive at scale.
- **Model-parallel training** — RMSNorm avoids the all-reduce sync
  for the per-shard mean.
- **Inference latency** matters — shaves per-block compute.

## When NOT to use

- **When mean-subtract is architecturally required** (e.g. some
  ResNet variants).
- **BatchNorm-style workflows** where per-batch stats are the
  point.
- **Legacy models** already trained with LayerNorm — dropping in
  RMSNorm requires retraining.

## Assumptions & caveats

- **No bias term** — mimics the paper's simplification; some
  impls add a learnable bias back.
- **Same γ semantics** as LayerNorm (per-channel scale).
- **Numerical eps** typically 1e-6; smaller = more precise, more
  prone to instability at FP16.
- **Equivalent quality** shown in original paper for MT, ASR, LMs.

## Related in this repo

- `dropout-batchnorm`, `spectral-normalization` — related
  normalisation techniques.
- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder`, `flash-attention`,
  `grouped-query-attention` — architectures pairing with RMSNorm.
- `weight-standardization` — weight-side normalisation cousin.
- `alibi-linear-attention-bias`, `rope-rotary-position-embedding`
  — LLaMA-family siblings.

## Run

```
python techniques/rmsnorm-normalization/python/rmsnorm_normalization.py
Rscript techniques/rmsnorm-normalization/r/rmsnorm_normalization.R
```

**Refs:** Zhang, B. & Sennrich, R. "Root mean square layer normalization." *NeurIPS*, 2019; Ba, J. L., Kiros, J. R. & Hinton, G. E. "Layer normalization." *arXiv:1607.06450*, 2016.

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
