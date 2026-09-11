# Medusa — Speculative Decoding with Extra Heads (§47.185)

Cai, Li, Geng, Peng, Lee, Zhang & Dao (2024). Instead of a
separate DRAFT MODEL (standard speculative decoding), Medusa adds
K extra **prediction heads** on top of the target LM:

    head_1 → token t+1   (baseline LM head)
    head_2 → token t+2   (speculative)
    …
    head_K → token t+K

Each head samples top-k candidates; a **tree-structured attention
mask** verifies many candidates in parallel with a single target-
model forward. Accepted prefix = longest matching path.

## Files

- `python/medusa_speculative_heads.py` — from-scratch multi-head
  verify loop on a synthetic distribution: 2 Medusa heads, top-k
  = 4, threshold = 0.3. Average tokens accepted per step:
  **2.58** (vs 1 baseline autoregressive) — **2.58× speed-up**.
- `r/medusa_speculative_heads.R` — no R port; recommends
  `FasterDecoding/Medusa`, `FastChat` integration.

## When to use

- **Small-batch, single-user** LLM inference — throughput-critical
  chat.
- **Draft-model-free** speculative decoding — no second small model
  to serve or align.
- **Existing base model** you can add heads to without retraining
  from scratch.

## When NOT to use

- **Very large batch sizes** — batch throughput already saturates
  the GPU; extra heads help less.
- **When a strong draft model exists** — separate draft (Chen 2023
  Assisted Generation) can be simpler.
- **Distribution mismatch**: heads mistrained on target output →
  low acceptance.

## Assumptions & caveats

- **Threshold** trades speed vs quality: lower = accept more risky
  drafts, higher = only very confident.
- **Head training** — freeze base LM, train heads on
  next-next-token prediction with ~few hours on 8 GPUs.
- **Tree width** grows exponentially with K; real impls use a
  hand-tuned sparse tree.
- **KV cache** for verify step needs care — same block of past K/V
  used by all speculated paths.

## Related in this repo

- `speculative-decoding` — standard draft-model variant.
- `beam-search-decoding` — orthogonal test-time search.
- `paged-attention-vllm`, `flash-attention` — serving-side
  neighbours.
- `tree-of-thoughts-reasoning` — different tree-structured
  reasoning idea.

## Run

```
python techniques/medusa-speculative-heads/python/medusa_speculative_heads.py
Rscript techniques/medusa-speculative-heads/r/medusa_speculative_heads.R
```

**Refs:** Cai, T. et al. "Medusa: Simple LLM inference acceleration framework with multiple decoding heads." *arXiv:2401.10774*, 2024; Leviathan, Y., Kalman, M. & Matias, Y. "Fast inference from transformers via speculative decoding." *ICML*, 2023.

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
