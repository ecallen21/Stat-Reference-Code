# GPTQ — Generative Pre-trained Transformer Quantization (§47.180)

Frantar, Ashkboos, Hoefler & Alistarh (2023, ICLR). Post-training
quantization guided by a small **calibration** dataset. For each
linear layer, minimises reconstruction error:

    Ŵ = argmin_{W′ ∈ Q}  ‖W X − W′ X‖_F²

Uses Optimal Brain Surgeon (Hassibi 1993) style **second-order
updates**: quantise columns one at a time, propagating the
compensation to the remaining columns via H⁻¹.

## Files

- `python/gptq_quantization.py` — from-scratch GPTQ INT4 with
  Cholesky-based Hessian inverse and column-wise OBS updates.
  Linear layer (d_in = 128, d_out = 64), 512-sample calibration:
  - Naive RTN INT4: **rel-err 0.162**.
  - **GPTQ INT4: rel-err 0.116** — 28 % of the naive error is
    removed.
  - Storage: **4 KB vs FP16 16 KB** (4× smaller).
- `r/gptq_quantization.R` — no R port; recommends `auto-gptq`,
  `IST-DASLab/gptq`, `optimum`.

## When to use

- **INT4 / INT3 LLM inference** — GPTQ is the standard "just
  quantize the weights" approach.
- **When you have a calibration set** (typically 128-512 samples
  of representative data).
- **Serving multiple copies** of a static model — one-time
  quant cost, cheap forever.

## When NOT to use

- **When you need to train / fine-tune** — quantised weights are
  not differentiable.
- **When per-sample calibration** is expensive (streaming data).
- **Very small linear layers** — the Hessian solve dominates
  quant time.

## Assumptions & caveats

- **Calibration data quality** matters — outlier / off-domain
  samples degrade quant.
- **Group size** (typically 128) trades scale-vector storage
  vs error.
- **Symmetric vs asymmetric** — GPTQ paper uses symmetric per-
  group; asymmetric can improve some layers.
- **Cholesky reordering** by descending H diag improves numerical
  stability.

## Related in this repo

- `awq-quantization` — activation-aware alternative to GPTQ.
- `bitsandbytes-int8-llm`, `product-quantization-pq`,
  `quantization-pruning` — quantisation neighbours.
- `lottery-ticket-hypothesis` — related sparsity method.
- `kv-cache-quantization` — inference-time cache quant cousin.

## Run

```
python techniques/gptq-quantization/python/gptq_quantization.py
Rscript techniques/gptq-quantization/r/gptq_quantization.R
```

**Refs:** Frantar, E., Ashkboos, S., Hoefler, T. & Alistarh, D. "GPTQ: Accurate post-training quantization for generative pre-trained transformers." *ICLR*, 2023; Hassibi, B. & Stork, D. "Second order derivatives for network pruning: Optimal brain surgeon." *NIPS*, 1993.

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
