# WGCNA — Weighted Gene Co-Expression Network Analysis (Reference §40.6)

Langfelder & Horvath (2008), Zhang & Horvath (2005). Turn an
expression matrix into a **weighted co-expression network** and cut
it into **modules** of tightly-coupled genes, then relate each
module to clinical traits via a **module eigengene** (first PC).

## Pipeline

1. Pairwise gene-gene Pearson correlation.
2. Soft-threshold: `a_ij = |cor(x_i, x_j)|^β` — β chosen for scale-
   free topology (typical 6-12).
3. Topological overlap matrix `TOM_ij = (l_ij + a_ij) / (min(k_i, k_j)
   + 1 − a_ij)`; use `1 − TOM` as dissimilarity.
4. Hierarchical clustering + Dynamic Tree Cut → modules.
5. **Module eigengene** = first PC of the module's expression;
   correlate with clinical traits.

## When to use

- **Bulk RNA-seq / microarray** — biological interpretation via
  co-expression structure.
- **Multi-omics integration** — pair module eigengenes with SNPs,
  methylation, or clinical outcomes.

## When NOT to use

- **Very small samples** (< 15) — network estimation is unstable.
- **Single-cell without adaptation** — use `hdWGCNA` variant.

## Files

- `python/wgcna_coexpression.py` — soft-threshold + TOM + average-
  linkage clustering with fixed-count cut + module eigengene. Demo
  (n=100 samples, p=60 genes = 3 true modules of 15 + 15 noise
  features): recovers a **36-gene module correlating −0.93** with
  the target trait built from factor 1.
- `r/wgcna_coexpression.R` — `WGCNA::blockwiseModules`,
  `WGCNA::moduleEigengenes`, `hdWGCNA` (R); `PyWGCNA`, custom
  (Python).

## Assumptions & caveats

- **β selection** — pick β so the network follows an approximate
  power law (`WGCNA::pickSoftThreshold`).
- **Batch effects** — remove before WGCNA (ComBat / SVA), else
  modules recover batch, not biology.
- **Module significance** — permutation-test module-trait
  correlations; correct for multiple modules.
- **Genes assigned to "grey"** — genes that don't fit any module
  are informative; do not drop.

## Related in this repo

- `differential-expression`, `gsea` — DE-first alternatives.
- `batch-effect-combat` — required preprocessing.
- `gaussian-graphical-model` — sparse-regression cousin.

## Run

```
python techniques/wgcna-coexpression/python/wgcna_coexpression.py
Rscript techniques/wgcna-coexpression/r/wgcna_coexpression.R
```

**Refs:** Langfelder, P. & Horvath, S. "WGCNA: an R package for weighted correlation network analysis." *BMC Bioinformatics*, 2008; Zhang, B. & Horvath, S. "A general framework for weighted gene co-expression network analysis." *Statistical Applications in Genetics and Molecular Biology*, 2005.

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
