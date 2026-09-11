# Autoformer — Series Decomposition Transformer (Reference §47.237)

Wu et al. (2021, NeurIPS). Innovations vs vanilla Transformer:

1. **Series Decomposition block**: x = trend + seasonal via
   moving average → trend, x − trend → seasonal. Applied inside
   every encoder / decoder layer.
2. **Auto-Correlation mechanism** replaces self-attention: match
   Q with K by TIME-DELAYED SIMILARITY (FFT-based).

Complexity O(L · log L). Yields strong long-horizon forecasts on
energy / weather / traffic benchmarks.

## Files

- `python/autoformer_decomposition.py` — moving-average
  decomposition + FFT auto-correlation on trend+seasonal series
  (T=300, backcast=100, horizon=40):
  - Detected dominant seasonal period: **25** (truth 24).
  - **Autoformer MSE = 1.42** vs naive **11.94** — 8× better.
- `r/autoformer_decomposition.R` — no R port; recommends
  `thuml/Autoformer`, `neuralforecast.Autoformer`.

## When to use

- **Strongly-seasonal** long-horizon series (energy, weather).
- **When trend + seasonal decomposition** is a natural mental
  model.
- **Efficiency-critical** — O(L log L) vs Transformer O(L²).

## When NOT to use

- **Non-seasonal series** — no auto-correlation peak to exploit.
- **Very short series** — decomposition needs enough history.

## Assumptions & caveats

- **Moving-average window** = seasonal period; sensitive to
  misspecification.
- **Auto-correlation lag range** — need to exclude trivial small
  lags to detect the actual period.
- **Decoder** uses cross-attention to encoder outputs; not just
  standalone series.

## Related in this repo

- `temporal-fusion-transformer`, `n-beats`,
  `deepar-probabilistic-forecast`, `informer-long-sequence`,
  `patch-tst`, `tsmixer-mlp-mixer` — sibling deep forecasters.
- `hodrick-prescott-filter`, `savitzky-golay-filter`,
  `functional-basis-smoothing` — decomposition cousins.

## Run

```
python techniques/autoformer-decomposition/python/autoformer_decomposition.py
Rscript techniques/autoformer-decomposition/r/autoformer_decomposition.R
```

**Refs:** Wu, H. et al. "Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting." *NeurIPS*, 2021.

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
