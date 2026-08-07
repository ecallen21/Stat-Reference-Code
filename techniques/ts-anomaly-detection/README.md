# Time Series Anomaly Detection (Reference §13.29)

Three complementary approaches, each best-suited to different signal structures.

## Hampel filter (rolling median + MAD)

Rolling window of half-width `k`. Robust center = median; robust scale = `1.4826 · MAD`. Flag `|y_t − median| > n · MAD`.

- **Best for**: series without strong trend or seasonality; robust to bursts of anomalies (unlike a rolling mean/sd, which is dragged by outliers).
- **Assumes**: local stationarity.

## STL-residual anomalies

Decompose `y_t = trend + seasonal + remainder` via **STL**. Flag remainder points whose magnitude exceeds `n × IQR`.

- **Best for**: signals with clear trend + seasonality (utility loads, sales, sensor drift).
- **Assumes**: known seasonal period; enough data for STL to fit.

## Predictive-residual (ARIMA / model-based) anomalies

Fit a predictive model (ARIMA, exponential smoothing, prophet-like), z-score the residuals, flag `|z| > n`.

- **Best for**: online monitoring with a well-tuned forecaster.
- **Assumes**: the model captures the normal dynamics; if it also fits the anomalies, they hide.

## Files

- `python/ts_anomaly_detection.py` — all three methods with a single synthetic demo (trend + 30-period seasonality + 5 injected extreme values). Hampel and STL each catch all 5 anomalies at precision ≈ 0.55 and 0.63; predictive-residual catches 4 of 5 at precision 0.44.
- `r/ts_anomaly_detection.R` — `pracma::hampel` and `forecast::tsoutliers`.

## Contrast with multivariate outliers

`multivariate-outlier-detection` (Batch 13) handles i.i.d. multivariate points with no time structure — Mahalanobis / MCD. Use it when observations are independent; use TS anomaly methods when time order matters.

## Reporting

- **Precision / recall** on labeled anomalies when ground truth exists.
- **F1** at a fixed threshold, or **AUC of the score** if you can rank without a hard cut.
- Prefer **score functions** (Hampel z, |residual|, ranking) over hard flags in production — thresholds are decisions for the operator.

## Assumptions & caveats

- **Choice of threshold** is a decision, not a statistic. `3σ` is standard; adjust with cost-of-error analysis.
- **Masking**: consecutive anomalies can hide each other in Hampel windows and in ARIMA fits — filter iteratively.
- **Concept drift**: rolling / online updates are safer for long deployment.

## Run

```
python techniques/ts-anomaly-detection/python/ts_anomaly_detection.py
Rscript techniques/ts-anomaly-detection/r/ts_anomaly_detection.R
```

**Refs:** Hampel, F.R. "The influence curve and its role in robust estimation." *JASA* 69(346), 383–393, 1974; Chen, C. & Liu, L.-M. "Joint estimation of model parameters and outlier effects in time series." *JASA* 88(421), 284–297, 1993.

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
