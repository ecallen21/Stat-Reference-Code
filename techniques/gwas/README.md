# GWAS (Reference §40.1, §40.23)

Uffelmann et al. (2021), Price et al. (2006). Genome-wide association
studies test millions of variants for association with a phenotype
under an additive genetic model (minor-allele count 0/1/2).

## Core outputs

- **Per-SNP β + SE + p** from linear or logistic regression on
  dosage.
- **Genome-wide significance** threshold `p < 5 × 10⁻⁸`
  (Bonferroni for ~10⁶ effective tests).
- **Genomic-control λ** `= median(χ²) / 0.4549`. λ > 1.05 flags
  population stratification or cryptic relatedness → add PCs or use
  a linear mixed model.
- **Manhattan + Q-Q** plots as visual summaries.

## Files

- `python/gwas.py` — vectorised per-SNP regression scan + genomic
  inflation λ. Demo (n=800, m=500, 3 true causal SNPs of small
  effect): λ = 0.89; top hit at SNP 50 with p = 1.4 × 10⁻¹⁵.
- `r/gwas.R` — `qqman`, `GWASTools`, `SNPassoc`, `GENESIS`,
  `bigsnpr` (R); `hail`, `pandas-plink`, `pysnptools` (Python);
  external: PLINK 2.0, REGENIE, SAIGE.

## When to use

- **Discovery** of trait-associated variants across genome-scale
  data.

## When NOT to use

- **Small samples** — GWAS needs 10³–10⁶ samples for realistic
  effect sizes.
- **Rare variants** — burden / SKAT tests instead of single-SNP
  tests.

## Assumptions & caveats

- **Population stratification** — cases and controls with
  different ancestry give false positives; correct with PCs
  (Price 2006) or LMMs (BOLT-LMM, SAIGE, REGENIE).
- **Additive model** — the workhorse; dominance / recessive tests
  add multiple-testing burden.
- **Multiple testing** — 5 × 10⁻⁸ is convention; adjust for
  imputation quality and effective number of tests.

## Related in this repo

- `polygenic-risk-scores` — aggregate weak-signal SNPs.
- `hardy-weinberg`, `linkage-disequilibrium`,
  `population-genetics-fst` — QC and population-structure tools.
- `mendelian-randomization` — use significant hits as instruments.
- `eqtl` — mechanism-of-action follow-up.

## Run

```
python techniques/gwas/python/gwas.py
Rscript techniques/gwas/r/gwas.R
```

**Refs:** Uffelmann, E., Huang, Q.Q., Munung, N.S. et al. "Genome-wide association studies." *Nature Reviews Methods Primers*, 2021; Price, A.L., Patterson, N.J., Plenge, R.M. et al. "Principal components analysis corrects for stratification in genome-wide association studies." *Nature Genetics*, 2006.

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
