# Gradient Checkpointing (Reference §47.169)

Chen, Xu, Zhang & Guestrin (2016). Standard backprop stores every
layer's activations to reuse them in the backward pass — O(L)
memory in depth L. Gradient checkpointing keeps only a **subset**
of activations ("checkpoints") and **recomputes** the rest during
backward:

    memory  = O(√L)     (equal-spaced checkpoints)
    compute ≈ 1.33 × forward pass  (one recomputation per segment)

Trades compute for memory; standard for training large
Transformers, Diffusion models, and long-sequence RNNs.

## Files

- `python/gradient_checkpointing.py` — from-scratch full-backprop
  and checkpointed backprop on a 25-layer tanh MLP:
  - Full stores **26 activations**.
  - **Checkpointed stores 6 activations** (segment = 5) — 4.3×
    memory reduction.
  - **Gradient discrepancy = 0.00e+00** — checkpointing is
    numerically exact.
- `r/gradient_checkpointing.R` — no native R port; recommends
  `torch.utils.checkpoint`, `transformers.PreTrainedModel.
  gradient_checkpointing_enable()`, `jax.checkpoint`.

## When to use

- **Any training that runs out of GPU memory** — the standard
  first-line fix.
- **Very deep or very-long-sequence models** — Transformers,
  RNNs, Neural ODEs.
- **Large batch sizes** in memory-tight settings.

## When NOT to use

- **When memory is plentiful** — the compute overhead is pure
  waste.
- **Non-deterministic layers** (dropout, BN with running-stat
  updates) need care — you must reseed / freeze the RNG in the
  recomputed segment.
- **When gradient accuracy is paramount** — although exact in
  theory, differentiating twice through non-linearities can
  amplify numerical error slightly.

## Assumptions & caveats

- **Segmentation** — equal-spaced O(√L) segments minimise
  worst-case memory; other splits (root-N-heuristic) trade
  memory vs recomputation depth.
- **Randomness inside segments** — must be reseeded on
  recomputation to match the original forward.
- **Fused / kernel-level** checkpointing (`torch.utils.
  checkpoint(use_reentrant=False)`) integrates with FSDP /
  activation offloading.
- **Complementary to mixed-precision** — both are near-free wins.

## Related in this repo

- `mixed-precision-training` — sibling memory optimisation.
- `flash-attention` — Transformer-specific memory-aware attention
  kernel.
- `quantization-pruning`, `product-quantization-pq` — inference-
  side compression.
- `mixture-of-experts`, `grouped-query-attention` — related
  scale-up techniques.

## Run

```
python techniques/gradient-checkpointing/python/gradient_checkpointing.py
Rscript techniques/gradient-checkpointing/r/gradient_checkpointing.R
```

**Refs:** Chen, T., Xu, B., Zhang, C. & Guestrin, C. "Training deep nets with sublinear memory cost." *arXiv:1604.06174*, 2016; Kirisame, M. et al. "Dynamic tensor rematerialization." *ICLR*, 2021.

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
