# Model Monitoring Metrics (Reference Ch 32 MLOps)

**Rolling-window monitoring** of a deployed classifier with two alert
families:

- **Absolute** — fire when a rolling metric crosses a fixed operational
  threshold (`accuracy < 0.75`, `ECE > 0.30`).
- **Relative** — fire when the metric drifts beyond `k · σ` of its own
  EWMA baseline (Shewhart / EWMA control chart).

## Metrics tracked

- **Accuracy / F1** — needs eventual labels.
- **Brier score** — probability-quality proper score.
- **ECE** — calibration gap.
- **Log-loss / NLL** — the training loss reported over deployed data.
- **Latency** (see `inference-latency-profiling`).

## EWMA baseline (Roberts 1959)

```
μ_t   ← (1 − α) μ_{t-1} + α  x_t
σ²_t ← (1 − α) σ²_{t-1} + α (x_t − μ_t)²
alert if | x_t − μ_t | > k · σ_t   (k = 3 gives ~ 0.3 % false-alarm under IID normal)
```

Combines with any base metric: `x_t` = rolling accuracy, rolling log-loss,
rolling latency, etc.

## When to use

- **Any deployed classification / regression model** where labels
  eventually arrive.
- **Trigger for retraining** — alerts feed the CI/CD pipeline.
- **Regulatory sign-off** — most audit regimes want rolling-metric
  reports with alerts and remediation.

## When NOT to use

- **Very slow label arrival** — combine with `data-drift-detection` (no
  labels needed) or `concept-drift-adwin` (semi-supervised proxies).
- **Non-stationary business** (holiday seasons, campaigns) — expect
  alerts even without model degradation; use segmented baselines.

## Files

- `python/model_monitoring_metrics.py` — from-scratch `Monitor` with a
  rolling window (default 200), rolling `accuracy / Brier / ECE`, EWMA
  baseline on accuracy, and absolute + relative alert rules. Demo on a
  synthetic 2000-example stream with a mid-stream accuracy drop at
  `t = 1000`: **ABS alert at t = 1150 (latency 150, ≈ rolling-window
  buffer), REL alert at t = 1001 (latency 1)**.
- `r/model_monitoring_metrics.R` — `qcc` control charts (R);
  `evidently` / `whylogs` / `arize` / `seldon-alibi-detect` (Python).

## Assumptions & caveats

- **Window size** — small ⇒ noisy metric; large ⇒ slow to detect.
- **EWMA `α`** — small (0.02 – 0.05) for stable baselines; larger for
  fast-moving traffic.
- **False-alarm rate** — `k = 3` is roughly 0.3 % under IID normal;
  serial correlation in production usually pushes this higher.
- **Metric selection** — accuracy alone can hide calibration degradation
  and vice versa; monitor multiple metrics.
- **Absolute vs relative** — use both. Absolute catches known-bad
  performance; relative catches deviations from an OK-but-different
  baseline.
- **Metric latency** — rolling metrics lag by ~ window-size samples;
  event-based detectors (`concept-drift-adwin`) can react faster.

## Related in this repo

- `data-drift-detection` — label-free input monitoring.
- `concept-drift-adwin` — event-based drift detection.
- `bayesian-ab-testing`, `multi-armed-bandits` — decide when to
  promote the new model.
- `calibration-scaling` — post-hoc fix triggered by ECE alerts.
- `inference-latency-profiling` — the latency counterpart.

## Run

```
python techniques/model-monitoring-metrics/python/model_monitoring_metrics.py
Rscript techniques/model-monitoring-metrics/r/model_monitoring_metrics.R
```

**Refs:** Roberts, S.W. "Control-chart tests based on geometric moving averages (EWMA)." *Technometrics*, 1959; Shewhart, W. *Economic Control of Quality of Manufactured Product*, Van Nostrand, 1931.

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
