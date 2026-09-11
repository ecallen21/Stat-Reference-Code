# Mixed-Precision Training (Reference §47.168)

Micikevicius et al. (2018). Uses FP16 (or BF16) for matrix
multiplications and activations for 2-4× speedup on tensor-core
GPUs, while keeping:

1. **A FP32 master copy of weights** (updates and small values
   don't underflow).
2. **Loss scaling**: multiply loss by S before backward, then
   unscale gradients by 1/S — shifts small gradients into the
   FP16-representable range.

## Files

- `python/mixed_precision_training.py` — numpy-side demonstration
  of FP16 numerics and gradient scaling:
  - Matmul relative error: FP16 = **1.7e-3**, FP32 = 1.1e-6, FP64
    = 0.
  - FP16 max = 6.55e+04, min-normal = 6.10e-05.
  - Small-gradient underflow: 1e-8 → 0 in FP16 (below min-normal).
  - **Loss scaling S = 2¹⁵** recovers 1e-8 after the FP16 round-
    trip (unscaled = 9.997e-9).
- `r/mixed_precision_training.R` — no native R port; recommends
  `torch.amp`, `tensorflow.keras.mixed_precision`, JAX bfloat16.

## When to use

- **Any modern GPU training** — 2-4× speedup at ~no accuracy loss
  on most models.
- **Large model / large batch fitting into limited VRAM**.
- **Distributed training** — halved communication in gradient
  all-reduce.

## When NOT to use

- **Very small models** where FP32 compute is not the bottleneck.
- **Unstable training** (RL, some GANs) — the reduced precision
  can amplify instabilities.
- **When numerical accuracy is critical** (scientific-computing
  pipelines, sensitive metrics).

## Assumptions & caveats

- **Master FP32 weights** are essential — pure-FP16 weights lose
  small updates from Adam / SGD.
- **Dynamic loss scaling** (torch.amp GradScaler) — adjusts S if
  gradients overflow or underflow.
- **BF16 vs FP16** — BF16 has wider range but less mantissa;
  usually needs no loss scaling.
- **Some ops stay in FP32** — softmax, exp, sum reductions,
  layer norm — the autocast policy handles this.

## Related in this repo

- `quantization-pruning`, `product-quantization-pq` — inference-
  side compression cousins.
- `gradient-checkpointing` — memory-vs-compute trade sibling.
- `mixture-of-experts`, `flash-attention`,
  `grouped-query-attention` — related efficient-training
  techniques.

## Run

```
python techniques/mixed-precision-training/python/mixed_precision_training.py
Rscript techniques/mixed-precision-training/r/mixed_precision_training.R
```

**Refs:** Micikevicius, P. et al. "Mixed precision training." *ICLR*, 2018; Kalamkar, D. et al. "A study of BFLOAT16 for deep learning training." *arXiv:1905.12322*, 2019.

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
