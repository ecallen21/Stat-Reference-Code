# FlashAttention (Reference §47.38)

Dao, Fu, Ermon, Rudra & Ré (2022, NeurIPS). Reduces memory traffic
between HBM and on-chip SRAM by **tiling** the attention matrix and
computing softmax **incrementally** with the online-softmax trick.

## Sources of speedup

- Never materialise the full `N × N` attention matrix in HBM.
- Recompute attention during backward instead of caching.
- Per-tile softmax update with running `(m, l)` statistics.

Memory: `O(N)` instead of `O(N²)` for the attention step. Wall-clock:
2-4× on modern GPUs.

## Files

- `python/flash_attention.py` — online-softmax tiled attention
  from scratch (the numerical trick behind Flash). Demo (T=64,
  d=8, various block sizes): tiled output matches naive attention
  to machine precision (`≤ 5.6e-16`) — algorithmic correctness
  demonstrated; CUDA-level tiling drives the actual GPU wall-clock
  win.
- `r/flash_attention.R` — no R port; `flash-attn`, xFormers,
  `torch.nn.functional.scaled_dot_product_attention`, vLLM
  (Python).

## When to use

- **Any transformer training / inference on modern GPUs** —
  contemporary default; `torch.SDPA` may pick it automatically.
- **Long sequences** — attention memory is the main bottleneck;
  FlashAttention keeps it linear.
- **Serving large models** — FA2 / FA3 are core to vLLM, TGI, etc.

## When NOT to use

- **CPU inference** — FlashAttention is a CUDA kernel; benefit is
  GPU-specific.
- **Very short sequences** — overhead can dominate the gain.
- **Non-standard attention shapes** (relative-bias, ALiBi + local)
  may not be supported by all Flash variants.

## Assumptions & caveats

- **Exact attention** — no approximation; different arithmetic
  order can cause tiny numerical diffs.
- **Kernel dispatch** — torch SDPA transparently picks the fastest
  backend based on shape / dtype / attention mask.
- **Backward recomputation** — memory saving from not caching the
  full softmax matrix.
- **FlashAttention-2 / 3** — later versions add multi-query,
  causal-mask, and Hopper-GPU optimisations.

## Related in this repo

- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder`, `grouped-query-attention`,
  `rope-rotary-position-embedding` — attention family.
- `speculative-decoding`, `inference-latency-profiling` — serving
  cousins.

## Run

```
python techniques/flash-attention/python/flash_attention.py
Rscript techniques/flash-attention/r/flash_attention.R
```

**Refs:** Dao, T. et al. "FlashAttention: fast and memory-efficient exact attention with IO-awareness." *NeurIPS*, 2022; Dao, T. "FlashAttention-2: faster attention with better parallelism and work partitioning." *ICLR*, 2024.

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
