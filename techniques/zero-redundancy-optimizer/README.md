# ZeRO — Zero Redundancy Optimizer / DeepSpeed (Reference §47.178)

Rajbhandari, Rasley, Ruwase & He (2020, SC). Three stages of
sharding across N data-parallel ranks:

- **Stage 1**: shard OPTIMISER STATES (Adam m, v, FP32 master).
- **Stage 2**: also shard GRADIENTS.
- **Stage 3**: also shard PARAMETERS (equivalent to FSDP).

Rajbhandari (2021, ZeRO-Infinity): offload to CPU / NVMe for
trillion-parameter models on modest hardware. Each stage adds
comm cost, so real deployments mix ZeRO with tensor / pipeline
parallelism.

## Files

- `python/zero_redundancy_optimizer.py` — memory model for a
  30-B-param model, N = 16 ranks, Adam:
  - DDP: **480 GB / rank** (impossible without sharding).
  - ZeRO-1: 142.5 GB (70.3 % reduction).
  - ZeRO-2: 86.2 GB (82.0 %).
  - ZeRO-3: **30.0 GB (93.8 %)** — fits on H100 (80 GB).
  - ZeRO-3 + offload: **7.5 GB (98.4 %)** — fits on A100 40 GB.
- `r/zero_redundancy_optimizer.R` — no R port; recommends
  `deepspeed`, `torch.distributed.fsdp`, `accelerate`.

## When to use

- **DeepSpeed workflows** — canonical way to enable ZeRO stages.
- **Model too large for DDP** — start at ZeRO-1, escalate to -3.
- **ZeRO-Offload / ZeRO-Infinity** for LLM training on limited
  hardware.

## When NOT to use

- **Small models** — DDP is faster with no memory pressure.
- **When you're already using pure FSDP** (ZeRO-3) — no need to
  add DeepSpeed as a dependency.
- **When PyTorch-native paths suffice** (FSDP + accelerate).

## Assumptions & caveats

- **Stage 3 ≡ FSDP** — same memory footprint, different runtime.
- **Offload cost** — CPU / NVMe reads slow per-step time; only
  useful when memory is the hard constraint.
- **Comm scaling** — stage 3's all-gather is 2× the stage 2 traffic.
- **Combine with TP / PP** for trillion-parameter models — ZeRO
  alone hits diminishing returns beyond ~64 ranks.

## Related in this repo

- `fsdp-fully-sharded-data-parallel` — the PyTorch-native ZeRO-3
  equivalent.
- `tensor-parallelism`, `pipeline-parallelism` — sibling
  parallelism dimensions.
- `mixed-precision-training`, `gradient-checkpointing` —
  training-toolbox neighbours.
- `mixture-of-experts` — expert-parallelism cousin also often
  paired with ZeRO.

## Run

```
python techniques/zero-redundancy-optimizer/python/zero_redundancy_optimizer.py
Rscript techniques/zero-redundancy-optimizer/r/zero_redundancy_optimizer.R
```

**Refs:** Rajbhandari, S., Rasley, J., Ruwase, O. & He, Y. "ZeRO: Memory optimizations toward training trillion parameter models." *SC*, 2020; Rajbhandari, S., Ruwase, O., Rasley, J., Smith, S. & He, Y. "ZeRO-Infinity: Breaking the GPU memory wall for extreme scale deep learning." *SC*, 2021.

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
