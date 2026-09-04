# eQTL Analysis (Reference §40.16)

Shabalin (2012), GTEx Consortium (2020). An **eQTL** is a genetic
variant that explains variation in a gene's expression level. **Cis**
eQTLs act within ~1 Mb of the gene; **trans** eQTLs act more distally
(different chromosome or > 1 Mb away).

## Standard test

Per (SNP, gene) pair:

```
log(expr_g) ~ β · SNP_dosage + covariates + ε
```

MatrixEQTL vectorises this over millions of SNP × thousands of
genes; permutation p-values or BH-FDR / Storey q-values handle
multiple testing.

## Files

- `python/eqtl.py` — vectorised SNP × gene regression restricted to
  a ±3-SNP cis window. Demo (n=300 samples, m=40 SNPs, g=20 genes,
  3 planted truth pairs — 2 cis, 1 trans): recovers **SNP 10→gene 12
  (p=8e-10, q=0.000)** and **SNP 3→gene 5 (p=8e-6, q=0.001)**; the
  trans-eQTL (SNP 25→gene 18) is outside the cis window and
  correctly not tested (would be caught in a trans-scan).
- `r/eqtl.R` — `MatrixEQTL::Matrix_eQTL_main`, QTLtools external,
  `qvalue` (R); `tensorqtl`, `hail`, `pandas-plink` + `statsmodels`
  (Python).

## When to use

- **Mechanism** follow-up on GWAS hits — does the risk allele
  change expression of a nearby gene?
- **Tissue-specific** regulation studies (GTEx, eQTLGen).

## When NOT to use

- **Small `n`** (< 100) — power is weak per pair.
- **Population stratification** without covariates — as in GWAS,
  correct with PCs.

## Assumptions & caveats

- **Linear additive model** — dominance / recessive tests possible
  but rarely used.
- **Multiple testing** — millions of tests; BH or hierarchical
  correction (gene-level then SNP-level within gene).
- **Cis window** — 1 Mb is convention; larger windows increase
  discovery but weaken FDR.
- **Trans-eQTL** — much larger burden; permutation nulls or LD-
  aware corrections needed.
- **Batch and cell-composition** — always include as covariates.

## Related in this repo

- `gwas` — companion variant-trait scan.
- `differential-expression`, `batch-effect-combat` — expression
  preprocessing.
- `mendelian-randomization` — uses cis-eQTLs as instruments.

## Run

```
python techniques/eqtl/python/eqtl.py
Rscript techniques/eqtl/r/eqtl.R
```

**Refs:** Shabalin, A.A. "Matrix eQTL: ultra fast eQTL analysis via large matrix operations." *Bioinformatics*, 2012; GTEx Consortium. "The GTEx Consortium atlas of genetic regulatory effects across human tissues." *Science*, 2020.

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
