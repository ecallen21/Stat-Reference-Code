# Pipeline Parallelism / GPipe (Reference §47.176)

Huang et al. (2019). Splits a deep model into K **sequential
stages** on K devices; splits a mini-batch into M **micro-batches**
so multiple stages compute concurrently:

    F_1(mb) → F_2(mb) → … → F_K(mb) → B_K → … → B_1
    (M micro-batches keep all K devices busy after warm-up)

**Bubble ratio** ≈ (K − 1) / (M + K − 1); larger M → less
bubble. **1F1B** (one-forward-one-backward, PipeDream) reduces peak
activation memory vs pure GPipe.

## Files

- `python/pipeline_parallelism.py` — bubble-ratio calculator and
  ASCII schedule display:
  - K = 4: M = 4 → 42.9 % bubble; **M = 16 → 15.8 %**;
    M = 32 → 8.6 %.
  - K = 3, M = 3 schedule shows the diagonal fill / drain pattern.
  - K = 8 stages, M = 32 micro-batches: **82.1 % pipeline
    efficiency**.
- `r/pipeline_parallelism.R` — no R port; recommends
  `torch.distributed.pipeline`, `deepspeed` pipeline, `megatron-lm`.

## When to use

- **Very deep models** where a single GPU can't hold the whole
  network.
- **Cross-node** parallelism where TP's all-reduce is too slow —
  PP only sends stage-boundary activations.
- **Combined with TP + DP** as "3D parallelism" for LLM training.

## When NOT to use

- **Small mini-batches** — bubble dominates; M ≥ 4K needed for
  reasonable efficiency.
- **Fast intra-node interconnect** — TP is usually preferred at
  the same stage count.
- **Very variable stage compute** — mismatched stage FLOPs
  amplify the bubble.

## Assumptions & caveats

- **Stage balance** — carve up layers so each stage takes ~equal
  time; unbalanced stages are the biggest efficiency killer.
- **1F1B / interleaved 1F1B** (Narayanan 2019, Megatron 2021)
  cuts peak activation memory vs GPipe.
- **Gradient accumulation** across micro-batches replaces per-step
  gradients.
- **Checkpointing** (recomputation) combined with PP is standard
  for very deep models.

## Related in this repo

- `tensor-parallelism`, `fsdp-fully-sharded-data-parallel`,
  `zero-redundancy-optimizer` — sibling model-scaling techniques.
- `mixed-precision-training`, `gradient-checkpointing` —
  training-toolbox neighbours.
- `mixture-of-experts` — expert parallelism cousin.

## Run

```
python techniques/pipeline-parallelism/python/pipeline_parallelism.py
Rscript techniques/pipeline-parallelism/r/pipeline_parallelism.R
```

**Refs:** Huang, Y. et al. "GPipe: Efficient training of giant neural networks using pipeline parallelism." *NeurIPS*, 2019; Narayanan, D. et al. "PipeDream: Generalized pipeline parallelism for DNN training." *SOSP*, 2019; Narayanan, D. et al. "Efficient large-scale language model training on GPU clusters using Megatron-LM." *SC*, 2021.

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
