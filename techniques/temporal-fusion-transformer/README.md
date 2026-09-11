# TFT — Temporal Fusion Transformer (Reference §47.233)

Lim, Arik, Loeff & Pfister (2021, IJF). Combines:

- **Gated Residual Networks** (GRN) for feature selection.
- **Variable-Selection Network** (VSN) at each timestep for
  interpretable feature attribution.
- **LSTM local encoder** (short-term) + Transformer multi-head
  attention (long-term).
- **Quantile output layer** for probabilistic forecasts.

Handles static covariates, known-future inputs (weather forecasts,
holidays), and observed past inputs uniformly.

## Files

- `python/temporal_fusion_transformer.py` — toy VSN + quantile
  forecast on 4-feature synthetic series (T=200, horizon=50):
  - VSN correctly ranks feature attention: **sinusoid 0.46**,
    event 0.34, trend 0.04, noise 0.16.
  - Test pinball loss 0.04, 10-90 % PI coverage 1.00.
- `r/temporal_fusion_transformer.R` — no R port; recommends
  `pytorch-forecasting.TemporalFusionTransformer`, `darts`.

## When to use

- **Multi-horizon forecasting** with mixed feature types.
- **Interpretability** — VSN attention names the top drivers.
- **Uncertainty-aware** decisions (quantile intervals).

## When NOT to use

- **Univariate short series** — plain ETS / ARIMA suffices.
- **Very long horizons** (> 1000 steps) — Autoformer / Informer
  more efficient.

## Assumptions & caveats

- **Categorical / static / future / past** covariates need
  correct type-tagging.
- **Quantile output** requires tuning quantile grid + coverage
  calibration.
- **LSTM warm-up** period matters for stability.

## Related in this repo

- `n-beats`, `deepar-probabilistic-forecast`,
  `informer-long-sequence`, `autoformer-decomposition`,
  `patch-tst`, `tsmixer-mlp-mixer` — sibling deep forecasters.
- `bayesian-quantile-regression`,
  `additive-quantile-regression` — quantile-regression cousins.

## Run

```
python techniques/temporal-fusion-transformer/python/temporal_fusion_transformer.py
Rscript techniques/temporal-fusion-transformer/r/temporal_fusion_transformer.R
```

**Refs:** Lim, B., Arik, S. O., Loeff, N. & Pfister, T. "Temporal fusion transformers for interpretable multi-horizon time series forecasting." *International Journal of Forecasting* 37(4), 2021.

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
