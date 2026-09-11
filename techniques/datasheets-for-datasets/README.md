# Datasheets for Datasets (Reference §47.220)

Gebru et al. (2018, revised 2021 CACM). A structured document
accompanying every dataset that answers:

- **Motivation** — why was it collected? funded by whom?
- **Composition** — instances, features, splits, targets.
- **Collection** — when? how? any people involved?
- **Preprocessing** — cleaning, resampling, filtering.
- **Uses** — intended uses; PROHIBITED uses.
- **Distribution** — license; how to obtain?
- **Maintenance** — who maintains? how to report errors?

Standard practice for research releases + regulated production
pipelines (HIPAA, GDPR). Complements **Model Cards** (Mitchell
2019).

## Files

- `python/datasheets_for_datasets.py` — 7-section template +
  completeness checker:
  - Well-documented example (healthcare readmission): **0
    missing fields**.
  - Under-documented example: **23 missing fields**, including
    `motivation.purpose`, `composition.target`, etc.
- `r/datasheets_for_datasets.R` — pure-R template + checker
  (identical structure).

## When to use

- **Every dataset release** — a datasheet is standard practice.
- **Regulated domains** (healthcare, finance) — required for
  audit / compliance.
- **Research reproducibility** — enables downstream users to
  understand provenance and limitations.

## When NOT to use

- **Not applicable** — datasheets are documentation, not
  training code; they cost only author time.
- **When the dataset is truly private / one-off** — an internal
  README may suffice.
- **When the dataset is being deprecated** — mark as such and
  reference the successor.

## Assumptions & caveats

- **Effort is 1-4 pages** of markdown; template makes it fast.
- **Prohibited uses** must be explicit — implicit trust is
  fragile.
- **Update on every version bump** — outdated datasheets are
  worse than none.
- **Sensitive attributes** — flag them so users can audit
  fairness downstream.

## Related in this repo

- `model-cards` — sibling documentation for models.
- `reproducibility-seeds`,
  `model-lineage-provenance`,
  `experiment-tracking` — MLOps-side reproducibility
  neighbours.
- `demographic-parity`, `equalized-odds`,
  `disparate-impact` — fairness metrics that need
  sensitive-attribute info from the datasheet.

## Run

```
python techniques/datasheets-for-datasets/python/datasheets_for_datasets.py
Rscript techniques/datasheets-for-datasets/r/datasheets_for_datasets.R
```

**Refs:** Gebru, T. et al. "Datasheets for datasets." *CACM* 64(12), 2021 (arXiv 2018); Mitchell, M. et al. "Model cards for model reporting." *FAT**, 2019; MLCommons. "Croissant: A metadata format for ML-ready datasets." *NeurIPS Datasets*, 2024.

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
