# Model Registry / Versioning (Reference Ch 32 MLOps)

**Persistent, versioned store of promoted models** — the promotion,
staging, production, archived lifecycle that sits between experiment
tracking and serving.

## Model / version / stage

- **Model name** — a business-level identifier (`churn_model`).
- **Version** — semver (`MAJOR.MINOR.PATCH`) per registered model.
- **Stage** — `none → staging → production → archived`.
- **Lineage** — the registered version carries the training-run id +
  input-dataset hash so it can be reproduced.

## Standard operations

- `register(name, version, run_id, dataset_sha, metrics)`
- `transition(name, version, stage)` — promoting to `production`
  auto-archives the previous production version.
- `rollback(name, to_version)` — re-promotes an archived version to
  production (transition-shaped).
- `get_production(name)` — the version serving live traffic.
- `timeline(name)` — the immutable event log for audit.

## When to use

- **Any model whose swap requires review** — the registry provides
  the review artefact.
- **Multi-team ownership** — the registry is the shared source of
  truth for "which model is live".
- **Regulated compliance** — event timeline is auditable.

## When NOT to use

- **Single one-off model** — a filename is enough.
- **Rapid churn without review** — the registry adds friction; keep
  it for models with a governance requirement.

## Files

- `python/model_registry_versioning.py` — from-scratch `ModelRegistry`
  with per-model JSON persistence, semver-string versions, promote /
  demote / rollback, and an immutable event `timeline()`. Demo:
  register `v1.0.0`, `v1.1.0`, `v2.0.0`; promote v1.1.0; promote
  v2.0.0 (auto-archives v1.1.0); rollback to v1.1.0. Full audit trail
  printed.
- `r/model_registry_versioning.R` — MLflow R client / `vetiver + pins`;
  `mlflow`, `wandb Model Registry`, `sagemaker`, `vertex-ai` (Python /
  cloud).

## Assumptions & caveats

- **Stage semantics are convention** — MLflow permits at most one
  production version per model; the demo enforces that.
- **Rollback is `transition(to_version, 'production')`** — the previous
  production auto-archives; make sure to record `reason` for the audit
  trail.
- **Version conflicts** — the demo raises on duplicate version
  registration; production systems use monotonic auto-versioning.
- **Model bytes are not stored here** — the registry references the
  content-addressed artifact stored via `experiment-tracking`.
- **Staging traffic gate** — real registries pair with the serving
  layer's traffic router (see `canary-deployment`) for progressive
  rollout.

## Related in this repo

- `experiment-tracking` — the run source that registered models come from.
- `canary-deployment`, `shadow-deployment` — the traffic mechanisms
  that consume registered versions.
- `model-lineage-provenance` — extended lineage across data + code +
  model versions.
- `model-cards` — the human-readable documentation that ships with
  each registered version.

## Run

```
python techniques/model-registry-versioning/python/model_registry_versioning.py
Rscript techniques/model-registry-versioning/r/model_registry_versioning.R
```

**Refs:** Zaharia, M. et al. "Accelerating the machine-learning lifecycle with MLflow." *IEEE Data Engineering Bulletin*, 2018; Sculley, D. et al. "Hidden technical debt in ML systems." *NeurIPS*, 2015.

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
