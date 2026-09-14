# ROCKET — Random Convolutional Kernel Transform (Reference §47.291)

Dempster, Petitjean & Webb (2020). Time-series classifier that
convolves each series with a LARGE random pool of kernels
(typically 10 000), then aggregates each kernel's response via:

- `max` value
- `PPV` (proportion of positive values)

Yielding a 20 000-dim feature vector; a linear (ridge /
logistic) classifier fits on top. State-of-the-art accuracy at
a fraction of the cost of shapelet / HIVE-COTE ensembles.

## Files

- `python/rocket_random_conv_features.py` — Random-kernel
  transform with SHARED kernels across train + test.
  Demo: 2-class sine-frequency problem (period 1 vs 3),
  N=200, L=100, 200 kernels. Ridge on ROCKET features
  achieves 100% test accuracy vs the mean-amplitude
  baseline's 48%.
- `r/rocket_random_conv_features.R` — reticulate + `sktime`
  (R); `sktime.transformations.panel.rocket`, `tslearn`,
  MiniRocket / MultiRocket, from-scratch (Python).

## When to use

- **Univariate / multivariate time-series classification** —
  strong baseline, minutes to train.
- **Feature extractor** for downstream ridge / logistic
  regression.
- **When you want simplicity + accuracy** — outperforms most
  hand-crafted feature ensembles.

## When NOT to use

- **Very long series with limited memory** — 10 000 kernels ×
  long series can be RAM-heavy; use MiniRocket.
- **When interpretability matters** — random kernels are
  opaque; use shapelet transform instead.
- **Small n** — Ridge on 20 000 features overfits without
  enough training samples.

## Assumptions & caveats

- **Kernels fixed at train time** — reuse them at inference;
  RE-sampling defeats the pipeline.
- **PPV bias term** — Dempster's original paper uses random
  biases; the PPV proportion is threshold-sensitive.
- **Dilation range** — capped by series length; the code
  handles this.
- **MiniRocket / MultiRocket** offer deterministic kernels
  and faster training with similar accuracy.

## Related in this repo

- `shapelet-transform` — interpretable-features cousin.
- `matrix-profile-anomaly` — subsequence-based methods.
- `dynamic-time-warping` — elastic distance baseline.
- `random-fourier-features` — analogous random-feature idea
  for kernel methods.
- `arima`, `prophet-forecasting` — classical time-series
  companions.

## Run

```
python techniques/rocket-random-conv-features/python/rocket_random_conv_features.py
Rscript techniques/rocket-random-conv-features/r/rocket_random_conv_features.R
```

**Refs:** Dempster, A., Petitjean, F. and Webb, G.I. "ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels." *Data Min. Knowl. Discov.*, 34: 1454-1495, 2020.

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
