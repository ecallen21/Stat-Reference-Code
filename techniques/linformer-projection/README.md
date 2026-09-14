# Linformer Projection (Reference §47.386)

Wang, Li, Khabsa, Fang & Ma (2020). Project the SEQUENCE
dimension of `K` and `V` from `n` to `k ≪ n` using learned
low-rank projections `E_k, E_v ∈ ℝ^{n × k}`:

```
K_proj = E_kᵀ K     (k × d)
V_proj = E_vᵀ V     (k × d)
attention(Q, K_proj, V_proj)     # O(n · k) instead of O(n²)
```

For fixed `k`, complexity is LINEAR in `n`. Wang et al show
that self-attention matrices are approximately low-rank
(effective rank ≪ n), making the projection a good
approximation for sequences up to ~ 4 096 tokens.

## Files

- `python/linformer_projection.py` — n=512, d=32 with a
  **chunk-averaging projection** as a simple stand-in for
  Linformer's LEARNED `E_k, E_v` (random Gaussian projections
  fail on softmax attention; production Linformer trains E
  end-to-end). Chunk-avg with `k=16` gives 2.7 % relative
  error using only 3 % of the score pairs; `k=256` gets to
  2.0 %.
- `r/linformer_projection.R` — reticulate to `linformer-pytorch`
  (R); `linformer-pytorch`, `x_transformers.Linformer`,
  from-scratch (Python).

## When to use

- **Long sequences with low-rank attention** — text, code
  where semantic redundancy is high.
- **Fixed sequence-length inference** — the learned E is
  tied to a max-length; deploy for fixed `n_max`.
- **Memory-bound training** — `k = 256` for `n_max = 4096`
  cuts attention memory 16×.

## When NOT to use

- **Variable-length inputs** — E is fixed to `n_max`;
  extending `n` requires re-training.
- **Sparse-attention structure** — Longformer / BigBird are
  better when attention is peaked and local.
- **Very short sequences** — vanilla attention is fine.

## Assumptions & caveats

- **Low-rank approximation** — Linformer's paper shows
  softmax(QKᵀ) is empirically low-rank ~ 20-64 on natural
  language, so `k ≈ 128-256` suffices.
- **Learned vs random projections** — random Gaussians ≠
  learned; the demo uses a chunk-average pooling that
  respects the low-rank structure.
- **Positional structure** — LOSES exact position info in
  the projected keys; combine with RoPE / ALiBi at query
  side.
- **Sharing E across heads / layers** — practical variants
  share for memory efficiency.
- **Causal masking** — Linformer's projection breaks causality
  because `K_j` mixes future keys; use causal variants
  (Linear-Transformer, causal Performer) instead.

## Related in this repo

- `reformer-lsh-attention`, `longformer-sparse-attention`,
  `performer-random-features`, `nystromformer-approximation`
  — efficient-attention siblings.
- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder`, `flash-attention`,
  `sliding-window-attention` — attention neighbours.
- `randomized-svd`, `nystrom-approximation`,
  `random-projections` — low-rank-approximation cousins.

## Run

```
python techniques/linformer-projection/python/linformer_projection.py
Rscript techniques/linformer-projection/r/linformer_projection.R
```

**Refs:** Wang, S., Li, B.Z., Khabsa, M., Fang, H. and Ma, H. "Linformer: self-attention with linear complexity." arXiv:2006.04768, 2020.

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
