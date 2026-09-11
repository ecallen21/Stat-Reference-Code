# FSDP — Fully Sharded Data Parallel (Reference §47.177)

Zhao et al. (2023, VLDB). Shards **parameters, gradients, AND
optimiser states** across N data-parallel ranks. Each rank holds
1/N of each tensor at rest; all-gathers just-in-time for compute:

    forward layer L:  all-gather params → compute → discard
    backward layer L: all-gather params → compute grad →
                       reduce-scatter grad → discard

Compared to DDP (each rank keeps a full weight copy): FSDP cuts
memory by N and matches DDP in throughput on modern hardware with
NVLink / NVSwitch.

## Files

- `python/fsdp_fully_sharded_data_parallel.py` — memory model
  for a 7-B-param LLM (LLaMA-7B-ish) with Adam optim states,
  N = 8 ranks:
  - **DDP: 112 GB / rank** (impossible on a 40 GB GPU).
  - ZeRO stage 1: 38.5 GB (65.6 % reduction).
  - ZeRO stage 2: 26.2 GB (76.6 %).
  - **FSDP / ZeRO-3: 14.0 GB (87.5 %)** — fits comfortably.
  - Max fittable in 40 GB / 8 ranks: DDP 2.5 B, ZeRO-1 7.3 B,
    ZeRO-2 10.7 B, **FSDP 20 B**.
- `r/fsdp_fully_sharded_data_parallel.R` — no R port; recommends
  `torch.distributed.fsdp`, `accelerate`, `deepspeed`.

## When to use

- **Training LLMs (7 B - 70 B+)** on 8-64 A100/H100 clusters.
- **Any scale where DDP's memory footprint doesn't fit** — FSDP
  is the standard fix.
- **Hybrid FSDP + PP + TP** as "3D parallelism" for trillion-
  parameter models.

## When NOT to use

- **Small models** (< 1 B params) — FSDP overhead exceeds
  benefit vs DDP.
- **Cross-node with slow interconnect** — the all-gather traffic
  becomes the bottleneck.
- **When gradient synchronisation must be exact-DDP** — FSDP's
  reduce-scatter can differ by summation order in some kernels.

## Assumptions & caveats

- **Sharding strategy** — FULL_SHARD (ZeRO-3), SHARD_GRAD_OP
  (ZeRO-2), NO_SHARD (DDP). Trade memory vs. comm.
- **CPU offload** — additional memory savings; slower per step.
- **Mixed precision** essential — FP16 / BF16 halves the sharded
  memory too.
- **Activation checkpointing** commonly paired with FSDP for very
  deep models.

## Related in this repo

- `zero-redundancy-optimizer` — the DeepSpeed formulation
  (ZeRO-3 = FSDP).
- `tensor-parallelism`, `pipeline-parallelism` — orthogonal
  parallelism dimensions.
- `mixed-precision-training`, `gradient-checkpointing` —
  training-toolbox neighbours.

## Run

```
python techniques/fsdp-fully-sharded-data-parallel/python/fsdp_fully_sharded_data_parallel.py
Rscript techniques/fsdp-fully-sharded-data-parallel/r/fsdp_fully_sharded_data_parallel.R
```

**Refs:** Zhao, Y. et al. "PyTorch FSDP: Experiences on scaling fully sharded data parallel." *VLDB*, 2023; Rajbhandari, S. et al. "ZeRO: Memory optimizations toward training trillion parameter models." *SC*, 2020.

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
