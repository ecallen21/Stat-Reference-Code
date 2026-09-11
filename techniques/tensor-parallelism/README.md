# Tensor Parallelism / Megatron-LM (Reference §47.175)

Shoeybi et al. (2020). Shards individual weight matrices ACROSS
devices so a single forward / backward requires just **two
all-reduce collectives per transformer block**:

    Column-parallel linear: W = [W_1, W_2, …] split by columns
        Y_i = X @ W_i   (each device); no all-reduce until later.
    Row-parallel linear:    W = [W_1; W_2; …] split by rows
        Y = Σ_i (X_i @ W_i)   (all-reduce sum).

Transformer block: QKV (col-parallel) → attention (per-head local)
→ output-proj (row-parallel; needs all-reduce) → MLP (col + row,
one all-reduce). Two all-reduces per layer forward + two per
backward.

## Files

- `python/tensor_parallelism.py` — simulated TP block with
  column / row-parallel linears. Runs at 1 / 2 / 4 shards on a
  32-D model:
  - **TP output is bit-exact vs single-device** (rel-err ≤ 4e-16).
  - Memory per shard scales as 1 / n_shards (80 KB → 40 KB → 20 KB).
- `r/tensor_parallelism.R` — no R port; recommends `megatron-lm`,
  `colossalai`, `accelerate`.

## When to use

- **Very large single-block layers** (Transformer FFN dim ≥ 8k) —
  weight matrices don't fit on one GPU.
- **Fast intra-node communication** (NVLink / NVSwitch) — TP's
  all-reduce is bandwidth-bound.
- **Combined with pipeline + data parallelism** as "3D parallelism"
  for trillion-parameter models.

## When NOT to use

- **Small models** where a single GPU has spare capacity — TP
  overhead > benefit.
- **Cross-node** communication — all-reduce over slow interconnect
  kills throughput.
- **Sequence parallelism / expert parallelism** may be better for
  specific bottlenecks.

## Assumptions & caveats

- **Two all-reduces per layer** are fundamental — designing away
  either requires activation redistribution.
- **Attention heads split cleanly across shards**; batch size
  must be divisible by n_shards for some variants.
- **Sequence parallelism** (Korthikanti 2023) additionally splits
  activations for LayerNorm / dropout — orthogonal to TP.
- **Dropout / BN** require identical randomness across shards
  (broadcast seed).

## Related in this repo

- `pipeline-parallelism`, `fsdp-fully-sharded-data-parallel`,
  `zero-redundancy-optimizer` — sibling large-model training
  techniques.
- `mixed-precision-training`, `gradient-checkpointing` —
  training-toolbox neighbours.
- `flash-attention`, `grouped-query-attention` — compute-side
  efficiency cousins.

## Run

```
python techniques/tensor-parallelism/python/tensor_parallelism.py
Rscript techniques/tensor-parallelism/r/tensor_parallelism.R
```

**Refs:** Shoeybi, M. et al. "Megatron-LM: Training multi-billion parameter language models using model parallelism." *arXiv:1909.08053*, 2020; Korthikanti, V. et al. "Reducing activation recomputation in large transformer models." *MLSys*, 2023.

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
