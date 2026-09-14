# Longformer Sparse Attention (Reference §47.385)

Beltagy, Peters & Cohan (2020). Restrict attention to a
SLIDING WINDOW plus a small set of GLOBAL tokens that
attend to (and are attended by) EVERYTHING:

```
for each token i: attend to i − w/2 … i + w/2   (local)
for each global g: attend to all tokens         (bridge)
                    all tokens attend to g
```

Complexity `O(n · w + n · G)` with `G` global tokens.
Enables sequences up to 4 096 tokens on standard GPUs; 16-8×
faster than quadratic attention with minimal accuracy loss.

## Files

- `python/longformer_sparse_attention.py` — `n=512`, `d=32`,
  tied Q=K. Window 16 gives relative output error 3.2 %
  using only 3.1 % of the score pairs; window 128 reaches
  2.0 % error at 25 % of pairs. Global tokens (4 evenly-
  spaced) barely move the error but bridge distant tokens
  — matters more on longer sequences.
- `r/longformer_sparse_attention.R` — no first-class R
  package; reticulate to `transformers.LongformerModel` (R);
  `transformers.LongformerModel`, `allenai/longformer`,
  from-scratch (Python).

## When to use

- **Document-level NLP** — long-answer QA, coreference,
  summarisation of full papers.
- **Sequences 1024-4096 tokens** — Longformer's sweet spot.
- **When some tokens are semantically "global" (CLS, SEP,
  headers)** — mark them as global to receive attention
  from everywhere.

## When NOT to use

- **Sequences ≤ 512** — vanilla attention is fine.
- **When every pair matters** — genome-level distant
  interactions may need Longformer + dilated windows.
- **Extremely long (≥ 16 k)** — use BigBird / LongNet /
  Mamba.

## Assumptions & caveats

- **Window size** — 512 in original LF paper for text; task-
  dependent.
- **Global tokens** — 4-8 typical (CLS, task-token, answer-
  span markers).
- **Dilated windows** — Longformer's paper adds dilated
  sliding-window layers for wider effective receptive field.
- **Causal masking** — decoder-only variant needs half-window
  masking.
- **Custom CUDA kernels** — production `transformers.Longformer`
  uses banded-matrix kernels; from-scratch demo is O(n·w)
  with naive slicing.

## Related in this repo

- `reformer-lsh-attention`, `linformer-projection`,
  `performer-random-features`, `nystromformer-approximation`
  — efficient-attention siblings.
- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder`, `flash-attention`,
  `sliding-window-attention`, `mamba-state-space-transformer`
  — attention neighbours.
- `sparse-transformer` (in `sparse-attention` if present) —
  earlier sparse-attention design.

## Run

```
python techniques/longformer-sparse-attention/python/longformer_sparse_attention.py
Rscript techniques/longformer-sparse-attention/r/longformer_sparse_attention.R
```

**Refs:** Beltagy, I., Peters, M.E. and Cohan, A. "Longformer: the long-document transformer." arXiv:2004.05150, 2020.

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
