# TSMixer — Time-Series MLP-Mixer (Reference §47.239)

Chen, Ekambaram, Nguyen, Chan, Vahid & Bengio (2023, TMLR). Adapts
MLP-Mixer (Tolstikhin 2021) to time series: alternates:

- **TIME-mixing MLP**: (T × C) → MLP across T dim.
- **FEATURE-mixing MLP**: (T × C) → MLP across C dim.

Simple + fast + strong. Beats Transformers on many long-horizon
benchmarks with a fraction of the parameters.

## Files

- `python/tsmixer_mlp_mixer.py` — 2-block time-mix + feature-mix
  demonstration with LS-fit output layer (T=100, C=3,
  horizon=20). Illustrates the alternating-mixer pipeline.
- `r/tsmixer_mlp_mixer.R` — no R port; recommends
  `google-research/tsmixer`, `darts`.

## When to use

- **Multivariate long-horizon** forecasting.
- **Compute-constrained** — MLPs are much cheaper than
  Transformers.
- **When benchmarks show TSMixer wins** on your dataset class
  (ETT, Electricity).

## When NOT to use

- **Long input windows without patching** — TSMixer's time-MLP
  scales linearly in T.
- **Very high-C multivariate** — feature-mixing MLP grows in C.

## Assumptions & caveats

- **Fixed input / output shapes** required per model.
- **Residual connections** essential for depth.
- **LayerNorm** between mixers stabilises training.

## Related in this repo

- `temporal-fusion-transformer`, `n-beats`,
  `deepar-probabilistic-forecast`, `informer-long-sequence`,
  `autoformer-decomposition`, `patch-tst` — sibling deep
  forecasters.
- `mixture-of-experts`, `mixture-of-depths` — mixer / routing
  cousins.

## Run

```
python techniques/tsmixer-mlp-mixer/python/tsmixer_mlp_mixer.py
Rscript techniques/tsmixer-mlp-mixer/r/tsmixer_mlp_mixer.R
```

**Refs:** Chen, S. et al. "TSMixer: An all-MLP architecture for time series forecasting." *TMLR*, 2023; Tolstikhin, I. et al. "MLP-Mixer: An all-MLP architecture for vision." *NeurIPS*, 2021.

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
