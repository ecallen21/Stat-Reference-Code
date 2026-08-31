# Experiment Tracking (Reference Ch 32 MLOps)

**Log every training run's params, metrics, artifacts, and
environment** so that comparisons across dozens of experiments are
possible and any run can be re-executed with the same behaviour.

## Minimal object model (MLflow / W&B / Neptune)

- **Run** — one training execution; identified by a UUID + name.
- **Params** — hyperparameters + code version + seed + environment.
- **Metrics** — scalar time series (per-epoch loss) + final validation
  numbers.
- **Artifacts** — model checkpoints, plots, notebooks; stored under a
  **content hash** so identical outputs across runs collapse to one file.
- **Environment** — Python + package versions + OS + hardware; usually
  the `uv.lock` / `pip freeze` snapshot.

## When to use

- **Every training pipeline you run more than once** — even solo work.
- **Team collaboration** — the tracker replaces the "which folder is
  the good model?" chat.
- **Regulated / reproducible science** — auditable log of every run
  and its outputs.

## When NOT to use

- **One-shot exploratory scripts** — the overhead outweighs the value.
- **Local demos** — a spreadsheet is fine.

## Files

- `python/experiment_tracking.py` — from-scratch `ExperimentTracker` +
  `Run` with `log_params`, `log_metric` (jsonl append), `log_artifact`
  (SHA-256 content-addressed store), and `best_run(metric, minimise)`
  query. Demo: 3-run learning-rate sweep on a synthetic regression;
  best run identified by `val_loss` = 0.27 at `lr = 0.1`; artifacts
  stored under their content hash.
- `r/experiment_tracking.R` — `mlflow` R client / `vetiver + pins`;
  `mlflow`, `wandb`, `neptune`, `comet`, `aim`, `dvc` (Python).

## Assumptions & caveats

- **Content-addressed storage** — identical artifacts collapse to a
  single stored blob, but the *metadata* still records which run
  produced which hash.
- **Metric jsonl** — appended one row per metric event; downstream
  visualisation is a plain `pd.read_json(lines=True)`.
- **Environment capture** — the demo does not snapshot `pip freeze`;
  production trackers should record it verbatim.
- **Distributed runs** — MLflow / W&B parents-and-children work but
  add ceremony; keep the design flat when possible.
- **Storage cost** — models × epochs × sweeps → GB fast. Enforce a
  retention policy or content-hash dedupe.

## Related in this repo

- `model-registry-versioning` — the promotion / staging layer that sits
  on top of experiment tracking.
- `model-lineage-provenance` — links tracker artifacts to the data
  splits that produced them.
- `reproducibility-seeds` — the params that make runs deterministic.
- `feature-store` — the feature values a training run consumed.
- `hyperparameter-tuning` (if present) — the natural user of a tracker.

## Run

```
python techniques/experiment-tracking/python/experiment_tracking.py
Rscript techniques/experiment-tracking/r/experiment_tracking.R
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
