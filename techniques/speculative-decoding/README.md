# Speculative Decoding (Reference §47.21)

Leviathan, Kalman & Matias (2023). Lossless inference-acceleration
trick: a **small draft model** proposes `k` tokens, then the **large
target model** verifies them in a single batched forward pass.

## Recipe

1. Draft `k` tokens autoregressively with the small model.
2. Batch them through the target model → target probabilities.
3. Rejection-sample per token:
   - Accept `x` if `U ≤ p_target(x) / p_draft(x)`.
   - On first rejection at position `j`, sample from residual
     `r(x) ∝ max(0, p_target(x) − p_draft(x))`, discard the rest.
4. If all `k` are accepted, take a bonus target sample "for free".

**Distribution-preserving** — output tokens are exactly distributed
as if sampled from the target model, one at a time.

## Speedup

Wall-clock drops by roughly `k · α / (1 + draft_cost)`, where `α` is
the acceptance rate. Well-aligned draft/target pairs achieve
2–5× speedups in practice (Chen et al. 2023, Medusa, EAGLE).

## Files

- `python/speculative_decoding.py` — from-scratch rejection-sampling
  loop over categorical distributions. Demo (V=32 vocab, 2000
  tokens, k=4): aligned draft (KL small) → 77% acceptance, 4.10×
  effective throughput; uniform-ish draft (KL big) → 28% acceptance,
  2.13× (worse, and overhead can wipe out the gain in real systems).
- `r/speculative_decoding.R` — no R implementations; describes vLLM,
  HF `assisted_generation`, Medusa, and EAGLE.

## When to use

- **Serving latency reduction** for autoregressive LMs.
- **Batch-1 inference** — the classic bottleneck; speculative gives
  the biggest win here.
- **A well-matched draft is available** — same tokeniser, similar
  distribution.

## When NOT to use

- **Poorly aligned draft** — overhead can EXCEED plain autoregression.
- **Very short outputs** — the constant setup cost dominates.
- **Extremely large batch sizes** — the target is already
  compute-bound; less to gain.

## Assumptions & caveats

- **Same tokeniser** for draft and target; token-level probabilities
  must match.
- **Correctness** — the rejection sampler EXACTLY preserves target
  distribution — no accuracy loss.
- **Non-greedy decoding** — the sampler covers `top-p`, `top-k` and
  temperature; greedy decoding uses a simple prefix-match rule.
- **Memory** — batched verification needs enough KV cache for `k`
  extra tokens.

## Related in this repo

- `text-generation-decoding` — the plain autoregressive baseline.
- `transformer-decoder`, `attention-mechanism` — the LM machinery.
- `knowledge-distillation` — training a good draft model
  (Medusa, EAGLE variants).
- `inference-latency-profiling` — how to measure the actual
  wall-clock win.

## Run

```
python techniques/speculative-decoding/python/speculative_decoding.py
Rscript techniques/speculative-decoding/r/speculative_decoding.R
```

**Refs:** Leviathan, Y., Kalman, M. & Matias, Y. "Fast inference from transformers via speculative decoding." *ICML*, 2023; Chen, C. et al. "Accelerating large language model decoding with speculative sampling." *arXiv:2302.01318*, 2023.

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
