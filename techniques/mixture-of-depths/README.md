# Mixture-of-Depths — MoD (Reference §47.194)

Raposo, Ainslie, Freitas, Neumann & Dean (2024, DeepMind). Like
Mixture-of-Experts routes across **width**, MoD routes across
**depth**:

    A per-token, per-block ROUTER decides whether the token passes
    through the block (full compute) or SKIPS it (identity).
    Router keeps top-k tokens per block; the rest are skipped.

Saves compute on 'easy' tokens (function words, repeated
punctuation) while spending it on 'hard' tokens (rare vocab,
reasoning steps).

## Files

- `python/mixture_of_depths.py` — 8-layer MLP-like Transformer
  block with per-block router:
  - Full-depth compute: **512 token-block operations**.
  - MoD (keep_frac = 0.5): **256 ops (50 % savings)**.
  - Output rel-diff vs full-depth: 0.62 (approximate — real MoD
    has the router trained end-to-end).
  - At T = 4096: keep_frac = 0.25 → 4× cheaper per layer.
- `r/mixture_of_depths.R` — no R port; recommends DeepMind's JAX
  reference or community PyTorch ports.

## When to use

- **LLM training with a fixed compute budget** — MoD can match
  a larger model's quality at fewer FLOPs.
- **Latency-sensitive inference** — router adaptively spends
  compute per token.
- **Combined with MoE** — MoDE (Mixture-of-Depths-and-Experts)
  in the paper.

## When NOT to use

- **When routing overhead dominates** — router forward + top-k is
  not free for small models.
- **Simple / uniform tasks** — everything's important; MoD saves
  nothing.
- **Deployment constraints** — variable per-token depth breaks
  some batching-optimised inference kernels.

## Assumptions & caveats

- **Router training** — end-to-end with a load-balancing loss;
  otherwise degenerates.
- **top-k selection** is straight-through estimator (STE) at
  training, hard-argmax at inference.
- **Deterministic compute budget** — keep_frac = k/T is fixed,
  not adaptive; MoDE relaxes this.
- **Not the same as early-exit** (Elhoushi 2024 LayerSkip) —
  which exits ALL tokens at the same layer.

## Related in this repo

- `mixture-of-experts` — sibling routing method (across width).
- `sparse-attention-*`, `flash-attention` — attention-efficiency
  cousins.
- `lottery-ticket-hypothesis`, `quantization-pruning`,
  `gptq-quantization`, `awq-quantization` — compression
  neighbours.

## Run

```
python techniques/mixture-of-depths/python/mixture_of_depths.py
Rscript techniques/mixture-of-depths/r/mixture_of_depths.R
```

**Refs:** Raposo, D. et al. "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models." *arXiv:2404.02258*, 2024; Elhoushi, M. et al. "LayerSkip: Enabling early exit inference and self-speculative decoding." *ACL*, 2024.

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
