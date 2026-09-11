# PagedAttention / vLLM (Reference §47.173)

Kwon et al. (2023, SOSP). LLM inference is memory-bound because the
KV cache grows with sequence length; naïve contiguous allocation
wastes memory to internal / external fragmentation.

    KV cache → BLOCKS (paged) instead of contiguous tensors.
    A per-sequence BLOCK TABLE maps logical positions → physical
    blocks (analogous to virtual memory paging).

Enables **prefix sharing** across requests, **copy-on-write forking**
for parallel sampling, and near-zero waste in KV memory.

## Files

- `python/paged_attention_vllm.py` — from-scratch block allocator +
  reference-counted paged KV cache with fork / free operations:
  - Seq A: 40 tokens → 3 blocks (0, 1, 2).
  - Fork A → B: shared 3 blocks, refs = [2, 2, 2].
  - B extends +20 tokens → allocates 2 new blocks (didn't touch
    shared).
  - Free A → refs decrement; blocks kept because B still holds
    them.
  - 60-token seq B: 5 blocks × 16 slots (25 % internal
    fragmentation) but **shares 3 blocks with siblings**.
- `r/paged_attention_vllm.R` — no R port; recommends `vllm`,
  `text-generation-inference`, `TensorRT-LLM`.

## When to use

- **LLM inference serving** — vLLM's biggest single throughput win.
- **Beam search / parallel sampling** — copy-on-write dramatically
  cuts memory.
- **High-throughput chat servers** with many concurrent sessions
  sharing prompt prefixes.

## When NOT to use

- **Single-user, single-sequence** inference — the overhead of a
  block table costs more than it saves.
- **Very short prompts / no sharing potential**.
- **Custom accelerators** (TPU / IPU) — may need a different
  paging strategy.

## Assumptions & caveats

- **Block size** trade-off — larger = less table overhead, more
  internal frag; typical 16-32 tokens.
- **Block sharing** requires content equality (prompt prefixes);
  hash-lookup on the block level.
- **Copy-on-write** — write to a shared block first copies it out
  to a private one (refcount decrement + fresh allocation).
- **Not free** — extra pointer chasing costs a few percent latency
  vs contiguous.

## Related in this repo

- `flash-attention`, `grouped-query-attention` —
  attention-kernel efficiency neighbours.
- `speculative-decoding` — orthogonal LLM inference speedup.
- `kv-cache-quantization` — combines with paging for even lower
  memory.
- `tensor-parallelism`, `pipeline-parallelism`,
  `fsdp-fully-sharded-data-parallel` — sibling model-scaling
  techniques.

## Run

```
python techniques/paged-attention-vllm/python/paged_attention_vllm.py
Rscript techniques/paged-attention-vllm/r/paged_attention_vllm.R
```

**Refs:** Kwon, W. et al. "Efficient memory management for large language model serving with PagedAttention." *SOSP*, 2023; Yu, G.-I. et al. "Orca: A distributed serving system for transformer-based generative models." *OSDI*, 2022.

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
