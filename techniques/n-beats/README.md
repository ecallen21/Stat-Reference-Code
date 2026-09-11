# N-BEATS — Neural Basis Expansion (Reference §47.234)

Oreshkin, Carpov, Chapados & Bengio (2020, ICLR). Pure deep-
learning forecaster; NO recurrence, NO attention.

    Stack of BLOCKS, each fits a partial signal:
    - Backcast: reconstruct the past.
    - Forecast: predict the future.
    Residuals passed to the next block ('doubly residual').

Blocks use polynomial / seasonal basis functions → interpretable.
Won the M4 competition ahead of every classical baseline.

## Files

- `python/n_beats.py` — 2-block N-BEATS (linear-trend + 3-harmonic
  Fourier) on trend + seasonal series (T=200, horizon=20):
  - **N-BEATS MSE = 0.41** vs naive last-value **6.01** — 15×
    better.
- `r/n_beats.R` — no R port; recommends `darts.NBEATSModel`,
  `neuralforecast`.

## When to use

- **M-competition-style** univariate forecasts.
- **Interpretable** components (trend + seasonal blocks).
- **Fast inference** — one forward pass per horizon.

## When NOT to use

- **Multivariate covariates** — plain N-BEATS is univariate;
  use N-BEATS-X for exogenous features.
- **Very long horizons with weak trend** — Autoformer /
  PatchTST fit better.

## Assumptions & caveats

- **Interpretable-basis blocks** trade capacity for
  interpretability; generic (unconstrained) blocks perform
  slightly better.
- **Backcast + forecast length** should be balanced (~1:2).
- **Ensemble** across seeds boosts M4-style benchmarks.

## Related in this repo

- `temporal-fusion-transformer`,
  `deepar-probabilistic-forecast`,
  `informer-long-sequence`, `autoformer-decomposition`,
  `patch-tst`, `tsmixer-mlp-mixer` — sibling deep forecasters.

## Run

```
python techniques/n-beats/python/n_beats.py
Rscript techniques/n-beats/r/n_beats.R
```

**Refs:** Oreshkin, B. N., Carpov, D., Chapados, N. & Bengio, Y. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." *ICLR*, 2020.

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
