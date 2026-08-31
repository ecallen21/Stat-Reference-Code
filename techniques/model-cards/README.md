# Model Cards (Reference Ch 32 MLOps)

**Short, structured documentation** for a released ML model — Mitchell
et al. (2019) 'Model Cards for Model Reporting'. Standard fields make
comparisons across models trivial and force each release to explicitly
answer the "who / what / when / how / caveats" questions.

## Required sections (Mitchell 2019)

1. **Model details** — name, version, date, owner, licence.
2. **Intended use** — primary users, primary use, out-of-scope uses.
3. **Factors** — relevant demographic / environmental groups; not-
   evaluated groups declared explicitly.
4. **Metrics** — headline performance, calibration, thresholds.
5. **Evaluation data** — source, preprocessing, licence.
6. **Training data** — same fields.
7. **Quantitative analyses** — unitary + intersectional group results.
8. **Ethical considerations** — risks, mitigations, groups affected.
9. **Caveats and recommendations** — sensitivity to drift, known
   limitations, retraining cadence.

## When to use

- **Every released model** — pair with `model-registry-versioning`.
- **Regulatory / audit** — many jurisdictions now require documentation
  in a model-card style (EU AI Act 2024, NIST AI RMF).
- **Third-party sharing** — HuggingFace-style cards travel with the
  weights.

## When NOT to use

- **Never** — even a stub with `TBD` in most sections is better than
  no card.

## Files

- `python/model_cards.py` — `ModelCard` dataclass with the 9 required
  sections + `validate` (missing-section detector) + `to_markdown`
  formatter. Demo: full model card for a synthetic `churn_model
  v2.0.0`; validator on empty card reports the four leading missing
  sections.
- `r/model_cards.R` — `yaml` / `rmarkdown` / `vetiver` (R);
  `google model-cards-toolkit`, `huggingface ModelCard`, `mlflow`
  Description (Python).

## Assumptions & caveats

- **Fields are conventions, not schema** — the format has small
  variations (Google MCT vs HuggingFace vs internal); pick one and
  enforce it via `validate`.
- **Group evaluations require labelled group columns** — pair with
  `demographic-parity` / `equalized-odds` for automatic per-group
  metrics.
- **Cards can become stale** — pin the card version to the registered
  model version; regenerate every release.
- **Public sharing** — remove PII, dataset licences, internal notes
  before publication.
- **Datasheets for Datasets** (Gebru 2018) is the sibling document for
  the data side.

## Related in this repo

- `model-registry-versioning` — model cards ship with each registered
  version.
- `experiment-tracking` — training-run id in the card links back to
  the tracker.
- `demographic-parity`, `equalized-odds`, `calibration-parity` —
  fairness metrics reported in the card.
- `data-drift-detection`, `concept-drift-adwin` — the drift caveat
  section warns of when to retrain.

## Run

```
python techniques/model-cards/python/model_cards.py
Rscript techniques/model-cards/r/model_cards.R
```

**Refs:** Mitchell, M. et al. "Model cards for model reporting." *FAT\**, 2019; Gebru, T. et al. "Datasheets for datasets." *CACM*, 2021; EU Artificial Intelligence Act (2024) Art. 13 (transparency and provision of information to deployers).

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
