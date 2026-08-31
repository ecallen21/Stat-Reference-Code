# Shadow Deployment (Reference Ch 32 MLOps)

**Score every live request with both the production model and a
candidate model in parallel; only the production output is served to
users; the candidate's predictions are logged for offline analysis.**

The safest way to compare two models on real production traffic.

## Standard summaries

- **Agreement rate** — how often the two models produce the same
  prediction.
- **Prediction-probability gap** — mean `|p_prod − p_shadow|` on the
  chosen class.
- **Per-model accuracy / calibration** on the labelled subset (once
  labels arrive).
- **Slice-level disagreement** — feature-wise decomposition of where
  the two models diverge.

## When to use

- **Any candidate model** before it sees live traffic.
- **A / B setup impossible** (regulatory, contractual, tiny user base).
- **Novel model families** — a new architecture might behave
  unpredictably; shadow first.

## When NOT to use

- **Latency-critical services** — dual scoring doubles inference cost;
  batch-shadow only on sampled requests (5-10 %).
- **Stateful models** — session state must be tracked separately for
  each model to avoid cross-contamination.
- **Regulatory concerns about logging** — shadow predictions may be
  considered personal data.

## Files

- `python/shadow_deployment.py` — from-scratch `ShadowRouter` that
  dual-scores each request, logs a `ShadowRecord`, and reports
  agreement / probability gap / per-model accuracy / segment
  disagreement. Demo on synthetic three-feature stream: **prod
  accuracy 0.899 vs shadow 0.947** (candidate is materially better)
  with 91 % agreement and mean probability gap 0.14.
- `r/shadow_deployment.R` — `plumber` / `vetiver` (R); `seldon-core`,
  `kserve`, `bentoml`, `mlflow`, `ray-serve` (Python).

## Assumptions & caveats

- **Non-blocking** — the shadow call must not delay the prod response;
  fire-and-forget async on the shadow path.
- **Sampling** — for expensive shadows, log a 1-10 % sample instead
  of every request.
- **State isolation** — shadow must not share caches / rate-limits with
  prod, or its behaviour is contaminated.
- **Cost** — shadowing an LLM costs roughly the same as prod; budget for it.
- **Metric latency** — requires labels for per-model accuracy; without
  labels, only agreement + probability gap are available.

## Related in this repo

- `canary-deployment` — the natural next step after a shadow evaluation.
- `bayesian-ab-testing`, `multi-armed-bandits` — post-canary
  promotion decisions.
- `model-monitoring-metrics` — the metric machinery re-used for both
  models.
- `experiment-tracking`, `model-registry-versioning` — where the
  shadow comparison is recorded.

## Run

```
python techniques/shadow-deployment/python/shadow_deployment.py
Rscript techniques/shadow-deployment/r/shadow_deployment.R
```

**Refs:** Chip Huyen. *Designing Machine Learning Systems*, O'Reilly, 2022 (ch. 9: shadow / A-B / canary); Sculley, D. et al. "Hidden technical debt in machine learning systems." *NeurIPS*, 2015.

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
