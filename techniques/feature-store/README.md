# Feature Store (Reference Ch 32 MLOps)

**Serve the same feature values to training (batch, historical,
point-in-time correct) and to serving (real-time, low-latency,
current).** The single most-effective bulwark against train/serve skew.

## Standard abstractions

- **Entity** — a business identifier (user_id, transaction_id).
- **Feature view** — a named set of columns keyed by entity.
- **Offline store** — batch DB or lake (parquet, warehouse, DuckDB).
- **Online store** — key-value store for low-latency lookup (Redis,
  DynamoDB, Bigtable).
- **Materialisation** — sync offline → online, respecting `event_time`.
- **Point-in-time join** — fetch feature values *as of* a training
  label timestamp; never future-leak.

## When to use

- **Any team with more than one ML model** consuming shared features.
- **Real-time serving** — Redis-tier online store is essential when
  batch recomputation is too slow.
- **Auditability** — the store answers "what value did the model see
  at prediction time?" reliably.

## When NOT to use

- **Single batch-mode model** — a well-designed data-warehouse view is
  enough.
- **Very high write throughput** — the online store may be the
  bottleneck; consider streaming (Kafka) + local caches.
- **Complex, low-latency joins** in serving — feature stores handle
  KV lookups, not on-the-fly joins.

## Files

- `python/feature_store.py` — from-scratch in-memory `FeatureStore`
  with:
  - `write_offline(entity, event_time, features)`
  - `materialize()` — sync latest offline to online
  - `offline_get_features(entities, event_ts)` — point-in-time correct
  - `online_get_features(entities)` — latest values
  - `train_skew(offline, online)` — feature-by-feature audit
  Demo: point-in-time join returns historical values (user 42 @ ts=250
  → value from ts=200, not the future ts=300); skew detector catches a
  $ ↔ cents unit-conversion bug (diff = 2970).
- `r/feature_store.R` — `pins` / `vetiver` / `duckdb` (R); `feast`,
  `tecton`, `sagemaker-feature-store`, `vertex-ai-feature-store`
  (Python).

## Assumptions & caveats

- **event_time is precious** — always store the wall-clock at which
  the feature *value* became true, not the row-insert time.
- **Point-in-time correctness** — a naive join with `WHERE ts <= label_ts`
  can still leak if features are updated in-place. Prefer append-only
  event logs.
- **Materialisation freshness** — the SLA on `online_last_updated`
  must match the model's tolerance for stale features.
- **Backfill** — the offline store must be able to answer *historical*
  point-in-time queries for retraining; add snapshots or CDC (change
  data capture) logs.
- **Skew audit is a policy, not a package** — schedule feature-by-
  feature parity checks (`train_skew`) daily.

## Related in this repo

- `data-drift-detection` — detects when the input distribution shifts
  even without an obvious skew.
- `model-lineage-provenance` — the feature store is one leg of the
  full data → model → prediction lineage.
- `model-monitoring-metrics` — the metric machinery that alerts on
  train/serve skew consequences.
- `experiment-tracking`, `model-registry-versioning` — the surrounding
  MLOps stack.

## Run

```
python techniques/feature-store/python/feature_store.py
Rscript techniques/feature-store/r/feature_store.R
```

**Refs:** Chip Huyen. *Designing Machine Learning Systems*, O'Reilly, 2022 (ch. 7); Uber Michelangelo Palette (2017 engineering blog); Kleppmann, M. *Designing Data-Intensive Applications*, O'Reilly, 2017 (ch. 11 for stream / batch parity).

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
