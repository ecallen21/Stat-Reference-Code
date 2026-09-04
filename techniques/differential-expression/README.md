# Differential Expression Analysis (Reference §40.3)

Ritchie et al. (2015), Love-Huber-Anders (2014). Test which of
thousands of genes differ between conditions given only a few
replicates per group. Two paradigms:

- **Log-transformed expression** (microarray / voom-transformed
  RNA-seq): linear model + **moderated t** (limma's empirical-Bayes
  variance shrinkage).
- **RNA-seq counts**: **negative-binomial GLM** with dispersion
  shrinkage (DESeq2, edgeR).

Both borrow strength across genes to stabilise per-gene variance
estimates when `n_replicates` is small.

## Moderated t (limma)

```
s²_post = (d_0 · s²_0 + df · s²_g) / (d_0 + df)
t_mod   = (mean_diff) / √(s²_post · (1/n_a + 1/n_b))
```

Prior variance `s²_0` = average across genes; prior degrees `d_0`
tunes shrinkage strength.

## Files

- `python/differential_expression.py` — moderated t on log2
  expression + BH-FDR, compared to plain per-gene t. Demo (500
  genes, 5 vs 5, 30 true DE with lfc ±1.5): plain t + BH calls
  30 hits (TP=27, FP=3); **moderated t + BH calls 32 (TP=30, FP=2)** —
  small but consistent power gain.
- `r/differential_expression.R` — `limma::lmFit` + `eBayes`,
  `DESeq2`, `edgeR` (R); `pydeseq2`, `scanpy.tl.rank_genes_groups`,
  `diffxpy` (Python).

## When to use

- **Bulk RNA-seq / microarray** two- or multi-group comparison.
- **Single-cell DE** — use `scanpy.tl.rank_genes_groups` or
  `diffxpy` variants.

## When NOT to use

- **One replicate per condition** — no gene-wise variance;
  moderated t collapses to a fold-change filter.
- **Massive class imbalance** without care — DESeq2 / edgeR handle
  offsets and library-size normalisation; a naive linear model on
  raw counts does not.

## Assumptions & caveats

- **Normalisation** happens before DE — voom for counts, quantile
  or RUV for microarrays.
- **Multiple testing** — BH-FDR (Benjamini-Hochberg) is convention;
  q < 0.05 or 0.10 typical.
- **Batch effects** — include batch as a fixed effect or apply
  ComBat / SVA first (see `batch-effect-combat`).
- **Interpretability** — a "significant" gene with tiny fold change
  may be biologically irrelevant; report `log2FC` alongside p / q.

## Related in this repo

- `batch-effect-combat`, `wgcna-coexpression`, `gsea` — the DE
  workflow.
- `multiple-testing-corrections` — BH, BY, Storey q-values.
- `fdr-control` (if present) — general FDR primer.

## Run

```
python techniques/differential-expression/python/differential_expression.py
Rscript techniques/differential-expression/r/differential_expression.R
```

**Refs:** Ritchie, M.E., Phipson, B., Wu, D. et al. "limma powers differential expression analyses for RNA-sequencing and microarray studies." *Nucleic Acids Research*, 2015; Love, M.I., Huber, W., & Anders, S. "Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2." *Genome Biology*, 2014.

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
