# Informer — Long-Sequence Time-Series Forecasting (Reference §47.236)

Zhou et al. (2021, AAAI best paper). Three tricks to scale
Transformer forecasting to horizons in the thousands:

1. **ProbSparse attention**: for each query, keep only the top-u
   keys by dominance score (O(L · log L) instead of O(L²)).
2. **Self-attention distilling** between encoder layers halves T.
3. **Generative decoder** emits the whole forecast in ONE pass
   (no step-by-step autoregressive rollout).

Enables 720-step forecasts vs vanilla Transformer's 168.

## Files

- `python/informer_long_sequence.py` — ProbSparse attention +
  distilling demonstration at L=512, d=16:
  - Full attention: 262,144 ops per head.
  - **ProbSparse (u = log₂ L = 9): 4,608 ops per head (56×
    cheaper)**.
  - Distilling: 512 → 256 → 128 → 64 tokens across 3 layers.
- `r/informer_long_sequence.R` — no R port; recommends
  `zhouhaoyi/Informer2020`, `neuralforecast.Informer`.

## When to use

- **Long-horizon forecasts** where naive Transformer OOMs.
- **Univariate / multivariate** long-context problems.
- **Batch throughput** — generative decoder amortises well.

## When NOT to use

- **Short sequences** — no compute win.
- **When PatchTST / Autoformer already fit** — often simpler.
- **Very sparse / bursty** signals — ProbSparse can miss dominant
  keys.

## Assumptions & caveats

- **u ~ log L** is the paper's default; tune per task.
- **Distilling factor 2** per layer; deeper stacks compound
  reduction.
- **ProbSparse's dominance metric** trades exactness for O(L log L).

## Related in this repo

- `temporal-fusion-transformer`, `n-beats`,
  `deepar-probabilistic-forecast`, `autoformer-decomposition`,
  `patch-tst`, `tsmixer-mlp-mixer` — sibling deep forecasters.
- `flash-attention`, `sliding-window-attention` — attention-
  efficiency neighbours.

## Run

```
python techniques/informer-long-sequence/python/informer_long_sequence.py
Rscript techniques/informer-long-sequence/r/informer_long_sequence.R
```

**Refs:** Zhou, H. et al. "Informer: Beyond efficient transformer for long sequence time-series forecasting." *AAAI*, 2021.

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
