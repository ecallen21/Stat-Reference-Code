# LAMB — Layer-wise Adaptive Moments (Reference §47.356)

You et al. (2020). AdamW plus a per-LAYER TRUST RATIO that
rescales the update by `‖θ‖ / ‖r‖`, where `r` is the AdamW
step direction (with weight decay):

```
r = m̂ / (√v̂ + ε) + wd · θ
trust = clip(‖θ‖ / ‖r‖, low, high)
θ ← θ − η · trust · r
```

The layer-wise trust ratio makes the effective step
INVARIANT to the initial weight scale — critical for training
transformers with EXTREMELY LARGE mini-batches: 32 k for BERT,
64 k for ImageNet ResNet, up to 65 536 in the original paper.

## Files

- `python/lamb_optimizer.py` — Two-layer regression with
  weights at different scales (d₁=20, d₂=40, feature-B
  amplitude 5× feature-A). LAMB and AdamW reach comparable
  MSE (~0.03); differences show mostly at very-large-batch
  scale-invariance regimes that are hard to reproduce in a
  toy demo.
- `r/lamb_optimizer.R` — reticulate to `torch_optimizer.Lamb`
  (R); `torch_optimizer.Lamb`, `nvidia.LAMB`,
  `keras.optimizers.LAMB`, `deepspeed.FusedLamb`,
  from-scratch (Python).

## When to use

- **Extreme-scale distributed training** — global batch
  ≥ 8 000; single-GPU speedups are typically minor.
- **BERT / GPT-style pretraining** with data-parallel
  scaling.
- **When you want to remove per-layer LR warmup** — trust
  ratio absorbs some of that need.

## When NOT to use

- **Small models / small batches** — AdamW is simpler and
  equivalent.
- **When learning-rate schedules already handle scale
  imbalance** — LARS's simpler trust ratio is often enough.
- **Debugging** — LAMB adds two extra hyperparameters
  (trust-ratio clip range).

## Assumptions & caveats

- **Trust-ratio clipping** — practical implementations clip
  to `[1e-3, 10]` to prevent extreme steps early in training.
- **Group by parameter tensor** — the trust ratio is
  computed per parameter tensor, not per scalar.
- **Weight decay is INSIDE `r`** — the trust ratio scales
  both the gradient step and the decay together.
- **NVIDIA's Fused LAMB** — a well-tested reference; DeepSpeed
  Zero-1 wraps it for very-large-batch training.

## Related in this repo

- `adamw-decoupled-weight-decay`, `adam-optimizer`,
  `lion-optimizer`, `adafactor` — first-order neighbours.
- `zero-redundancy-optimizer`, `fsdp-fully-sharded-data-parallel`
  — distributed training the LAMB paper targets.
- `mixed-precision-training`, `gradient-checkpointing`,
  `pipeline-parallelism`, `tensor-parallelism` — scale-out
  siblings.
- `one-cycle-super-convergence`, `lr-schedules` — LR schedule
  cousins.

## Run

```
python techniques/lamb-optimizer/python/lamb_optimizer.py
Rscript techniques/lamb-optimizer/r/lamb_optimizer.R
```

**Refs:** You, Y., Li, J., Reddi, S., et al. "Large batch optimization for deep learning: training BERT in 76 minutes." In *ICLR*, 2020.

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
