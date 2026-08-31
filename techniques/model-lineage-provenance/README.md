# Model Lineage / Provenance (Reference Ch 32 MLOps)

Track the **DAG of everything** that went into a prediction:

```
RAW_DATA → FEATURE_VIEW → TRAINING_RUN → MODEL_VERSION → PREDICTION_BATCH
```

Every node carries a content **hash**; every edge records the
transformation. Answers ops questions like:

- Which prediction rows used feature-view `v3.1`?
- Which raw ETL sources fed `feature-view v3.1`?
- If a data-quality issue is found at the raw source, **which
  predictions must be re-scored** ("blast radius")?

## Two directions of query

- **Upstream** — root-cause. Given a bad prediction, walk parents to
  find the offending data / model / code node.
- **Downstream** — blast radius. Given a bad data source, walk
  children to enumerate every downstream artefact that must be
  re-produced.

## When to use

- **Any production ML system beyond one model** — lineage is what
  turns "we shipped it" into "we can explain it".
- **Regulated compliance** — data-provenance evidence is required by
  many privacy regimes (GDPR, HIPAA).
- **Incident response** — the DAG shortens the "what did this bug
  affect?" investigation from days to minutes.

## When NOT to use

- **Fully offline single-shot analysis** — lineage overhead outweighs
  the value.

## Files

- `python/model_lineage_provenance.py` — from-scratch `LineageGraph`
  with `add(kind, meta)`, `link(parent, child)`, `upstream(node)` and
  `downstream(node)`. Demo builds a
  raw→feature-view→run→model→prediction DAG and answers both queries:
  6-node upstream of a prediction batch; 5-node blast radius of a raw
  data source.
- `r/model_lineage_provenance.R` — `igraph` / `targets` (R);
  `openlineage`, `marquez`, `mlflow`, `dbt lineage`, `datahub`,
  `amundsen` (Python / data-platform ecosystem).

## Assumptions & caveats

- **Content hashes** — the DAG is only useful if node hashes match the
  actual bytes. Store the hash + confirm on re-read.
- **Granularity** — too coarse and blast-radius queries include
  irrelevant nodes; too fine and the graph explodes.
- **PROV standard** (W3C 2013) formalises the vocabulary; production
  lineage platforms usually emit PROV-compatible events.
- **Retention** — keep at least one year of lineage in production
  audits; longer for regulatory contexts.
- **Cycle safety** — a lineage graph must be acyclic; enforce on
  `link`.

## Related in this repo

- `experiment-tracking` — training-run node source.
- `model-registry-versioning` — model-version node source.
- `feature-store` — feature-view node source with point-in-time joins.
- `reproducibility-seeds` — content hashes travel with lineage nodes.
- `data-drift-detection`, `model-monitoring-metrics` — alerts feed the
  lineage query "what to re-score after drift".

## Run

```
python techniques/model-lineage-provenance/python/model_lineage_provenance.py
Rscript techniques/model-lineage-provenance/r/model_lineage_provenance.R
```

**Refs:** Missier, P. et al. "The W3C PROV family of specifications for modelling provenance metadata." *EDBT*, 2013; Sculley, D. et al. "Hidden technical debt in ML systems." *NeurIPS*, 2015; OpenLineage specification (open-lineage.io).

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
