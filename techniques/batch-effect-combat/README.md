# Batch-Effect Correction — ComBat (Reference §40.11, §40.14)

Johnson-Li-Rabinovic (2007). Systematic non-biological variation
across processing batches inflates false discoveries in downstream DE.
**ComBat** adjusts expression via an empirical-Bayes location-and-
scale model that preserves the biological design while removing
batch means and variances.

## Model

```
y_ijg = α_g + X_j · β_g + γ_ig + δ_ig · ε_ijg
```

- `γ_ig` — batch mean shift for gene `g`, batch `i`.
- `δ_ig` — batch variance scale.
- Both pooled across genes via an empirical-Bayes hyperprior so
  small-batch estimates borrow strength.
- `X_j · β_g` — biological covariates that must NOT be regressed
  out.

## When to use

- **Multi-site / multi-run** omics with known batch labels.
- **Cross-cohort integration** — merge studies with different
  processing pipelines.

## When NOT to use

- **Batch confounded with biology** — cannot separate the two; fix
  the experimental design.
- **Batch label unknown** — use SVA to estimate surrogate variables
  first.
- **Single-cell data** — prefer scaled MNN / Harmony / scVI-based
  integration.

## Files

- `python/batch_effect_combat.py` — parametric-prior ComBat with
  design-matrix protection of biology. Demo (n_genes=200, 3
  batches × 20 samples, batch means & scales injected, 20 truly DE
  genes): **cross-batch variance 0.70 → 0.04** after ComBat;
  biological LFC preserved at +1.12 (target 1.5).
- `r/batch_effect_combat.R` — `sva::ComBat` / `ComBat_seq`,
  `sva::sva`, `limma::removeBatchEffect`, `batchelor::fastMNN`,
  `harmony` (R); `neuroCombat`, `scanpy.pp.combat`, `harmonypy`,
  `scvi-tools` (Python).

## Assumptions & caveats

- **Batch not confounded with biology** — check with contingency
  table before running.
- **Include biology in the design** so it is not regressed out.
- **ComBat vs ComBat-seq** — use `ComBat_seq` on integer counts;
  `ComBat` on log-transformed data.
- **Downstream degrees of freedom** — ComBat consumes df; report
  the adjustment in any DE follow-up.

## Related in this repo

- `differential-expression`, `wgcna-coexpression` — downstream users
  of the corrected data.
- `data-drift-detection` — ML deployment cousin.
- `harmony` (if present) — single-cell integration.

## Run

```
python techniques/batch-effect-combat/python/batch_effect_combat.py
Rscript techniques/batch-effect-combat/r/batch_effect_combat.R
```

**Refs:** Johnson, W.E., Li, C., & Rabinovic, A. "Adjusting batch effects in microarray expression data using empirical Bayes methods." *Biostatistics*, 2007; Korsunsky, I., Millard, N., Fan, J. et al. "Fast, sensitive and accurate integration of single-cell data with Harmony." *Nature Methods*, 2019.

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
