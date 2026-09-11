# AWQ — Activation-aware Weight Quantization (Reference §47.181)

Lin et al. (2024, MLSys). Not all weights matter equally: those
multiplied by **large-magnitude activations** ("salient weights",
typically ~1 % of columns) drive most of the quantisation error.

    1. Compute per-channel activation statistics from calibration.
    2. SCALE UP salient weight columns W[:, j] by s_j (chosen so
       mean(|W|) ≈ mean(|s · W|)).
    3. Scale DOWN the corresponding input activations x_j / s_j
       (folded into upstream layer norm / linear).
    4. Quantise the rescaled weights.

Because post-rescale weights have smaller max / min, INT4 wastes
fewer levels on outliers → less quant error.

## Files

- `python/awq_quantization.py` — from-scratch AWQ with a small
  grid search over the salience exponent α. Linear layer (d_in
  = 128, d_out = 64), 5 injected salient input channels:
  - RTN INT4 (per-group=64): **rel-err 0.104**.
  - **AWQ INT4 (α* = 0.3): rel-err 0.081** — 22 % of RTN error
    removed.
- `r/awq_quantization.R` — no R port; recommends `llm-awq`,
  `autoawq`, `vllm.awq`.

## When to use

- **Serving quantised LLMs** — AWQ INT4 is a leading alternative
  to GPTQ, often 1-2× faster to compute the quant.
- **When activations have clear outliers** (typical of Transformer
  FFN / attention paths).
- **Kernel-friendly deployment** — AWQ pairs with marlin / awq
  fused INT4 × FP16 kernels.

## When NOT to use

- **When activation stats are hard to collect** — AWQ needs a
  small calibration set.
- **When you need INT8** — AWQ is designed for INT4; INT8 rarely
  benefits.
- **Very small models** — quant error is negligible with either
  method; simpler RTN is fine.

## Assumptions & caveats

- **Salience score** is per-input-channel activation magnitude,
  not per-weight — different from magnitude-pruning.
- **α (search range)** — 0.0-1.0 with a small grid; α = 0 is RTN,
  α = 1 fully absorbs the outlier magnitude.
- **Scale folding** into upstream ops adds zero runtime cost;
  requires the previous layer to accept a scaling term (LN,
  linear).
- **Combined with GPTQ** in some pipelines (best-of-both scale
  + Hessian correction).

## Related in this repo

- `gptq-quantization` — sibling PTQ method (Hessian-based).
- `bitsandbytes-int8-llm`, `product-quantization-pq`,
  `quantization-pruning`, `kv-cache-quantization` — quant
  neighbours.
- `lora-peft` — parameter-efficient fine-tuning cousin often
  paired with AWQ inference (e.g. QLoRA + AWQ).

## Run

```
python techniques/awq-quantization/python/awq_quantization.py
Rscript techniques/awq-quantization/r/awq_quantization.R
```

**Refs:** Lin, J. et al. "AWQ: Activation-aware weight quantization for LLM compression and acceleration." *MLSys*, 2024; Frantar, E. et al. "GPTQ: Accurate post-training quantization for generative pre-trained transformers." *ICLR*, 2023.

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
