# PatchTST — Patch Time-Series Transformer (Reference §47.238)

Nie, Nguyen, Sinthong & Kalagnanam (2023, ICLR). Two tricks:

1. **Patching**: split the series into non-overlapping patches
   (like ViT), then apply Transformer to the sequence of patch
   embeddings.
2. **Channel-independent**: each variate is forecast by the SAME
   shared Transformer → parameter-sharing bonus + no spurious
   cross-variate leakage.

Simple, strong long-horizon forecaster; beat Informer / Autoformer
on ETT / Weather / Electricity.

## Files

- `python/patch_tst.py` — patch + mean-pool head demonstration
  (T=400, patch=8, stride=8, horizon=20):
  - 6 patches from 48-step past window.
  - Toy mean-pool head is weaker than naive; the real
    contribution is the patching + channel-independent training.
- `r/patch_tst.R` — no R port; recommends `thuml/PatchTST`,
  `neuralforecast.PatchTST`, `darts`.

## When to use

- **Long-horizon univariate/multivariate** forecasting.
- **Faster training** than Informer / Autoformer with
  competitive accuracy.
- **When RevIN normalisation** is used per-series (paper trick).

## When NOT to use

- **Short horizons** — patching gains negligible.
- **Very small patch size** — degenerates to per-token
  Transformer.

## Assumptions & caveats

- **Patch length** = period / 2 typical; stride = patch_len
  (non-overlapping).
- **Channel-independence** means one shared Transformer over all
  variates; extension: PatchTST + cross-variate mixing.
- **RevIN** (Reversible Instance Normalisation, Kim 2022)
  usually paired.

## Related in this repo

- `temporal-fusion-transformer`, `n-beats`,
  `deepar-probabilistic-forecast`, `informer-long-sequence`,
  `autoformer-decomposition`, `tsmixer-mlp-mixer` — sibling
  deep forecasters.
- `vision-transformer-vit` — patch-tokenisation ancestor.

## Run

```
python techniques/patch-tst/python/patch_tst.py
Rscript techniques/patch-tst/r/patch_tst.R
```

**Refs:** Nie, Y., Nguyen, N. H., Sinthong, P. & Kalagnanam, J. "A time series is worth 64 words: Long-term forecasting with transformers (PatchTST)." *ICLR*, 2023.

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
