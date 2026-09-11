# DeepAR — Autoregressive Probabilistic Forecast (Reference §47.235)

Salinas, Flunkert, Gasthaus & Januschowski (2020, IJF). LSTM
produces the parameters of a per-timestep **distribution**
(Gaussian mean+var or Negative-Binomial (μ, α) for counts):

    h_t = LSTM(y_{t-1}, x_t, h_{t-1})
    θ_t = linear(h_t)
    y_t ~ Distribution(θ_t)

Train by maximising log-likelihood. Sample multi-horizon
predictive distribution by rolling out with Monte-Carlo draws.

## Files

- `python/deepar_probabilistic_forecast.py` — random-walk proxy
  demonstrating the MC-rollout + quantile-interval structure
  (real DeepAR uses trained LSTM):
  - 200 MC samples, horizon 20.
  - **5-95 % PI coverage = 0.90** (matches nominal exactly).
- `r/deepar_probabilistic_forecast.R` — no R port; recommends
  `gluonts.model.deepar`, `darts.DeepAR`, `pytorch-forecasting`.

## When to use

- **Probabilistic forecasting** where PI matters (inventory,
  capacity).
- **Cross-series learning** — DeepAR shares an LSTM across many
  time series.
- **Count / zero-inflated** targets — Negative-Binomial output
  head.

## When NOT to use

- **Very short series** — need enough history to fit LSTM.
- **When point forecasts suffice** — plain LSTM / GBM cheaper.

## Assumptions & caveats

- **Choice of output distribution** matters — Gaussian for
  continuous, NB for counts, StudentT for heavy-tailed.
- **MC sample budget** — 100-1000 typical; more = tighter PI
  estimates but slower.
- **Global vs local models** — DeepAR shares weights globally;
  entity-level embedding conditions on series identity.

## Related in this repo

- `temporal-fusion-transformer`, `n-beats`,
  `informer-long-sequence`, `autoformer-decomposition`,
  `patch-tst`, `tsmixer-mlp-mixer` — sibling deep forecasters.
- `bayesian-quantile-regression`,
  `additive-quantile-regression` — quantile-side cousins.

## Run

```
python techniques/deepar-probabilistic-forecast/python/deepar_probabilistic_forecast.py
Rscript techniques/deepar-probabilistic-forecast/r/deepar_probabilistic_forecast.R
```

**Refs:** Salinas, D., Flunkert, V., Gasthaus, J. & Januschowski, T. "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." *IJF* 36(3), 2020.

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
