# Haplotype Phasing (Reference §40.27)

Excoffier & Slatkin (1995), Stephens-Smith-Donnelly (2001). Diploid
genotypes are **unphased**: for a double-heterozygote we cannot tell
whether the two minor alleles sit on the same chromosome (cis) or
opposite chromosomes (trans). **Phasing** infers haplotypes from
the observed genotypes.

## EM (Excoffier-Slatkin 1995)

For 2 SNPs there are 4 haplotypes `{AB, Ab, aB, ab}` with frequencies
`(f₁, f₂, f₃, f₄)`.

- **E-step**: for each ambiguous individual, weight each compatible
  haplotype pair by `f_i · f_j` (normalised).
- **M-step**: haplotype frequencies = weighted counts / (2 · n).

Production tools (SHAPEIT, Beagle, Eagle) do sequential imputation
on reference panels of hundreds of thousands of samples.

## When to use

- **Haplotype-based association testing** — sometimes more powerful
  than single-SNP GWAS at complex loci (HLA).
- **Long-range LD analyses**, ancestry inference, IBD detection.
- **Imputation** of ungenotyped variants from a reference panel.

## When NOT to use

- **Analyses that only need dosages** — single-SNP GWAS never
  needs phase.
- **Complex trios / pedigrees** — Mendelian phasing via
  transmission is exact where the family structure allows.

## Files

- `python/haplotype_phasing.py` — 2-SNP EM (Excoffier-Slatkin) on
  n=500 unphased genotypes drawn from true haplotype frequencies
  `(0.4, 0.1, 0.1, 0.4)`. Demo: estimated `(0.369, 0.104, 0.112,
  0.415)` — recovers the LD structure (implied r² = 0.322).
- `r/haplotype_phasing.R` — `haplo.stats::haplo.em`,
  `gap::hap.em`, `SNPassoc` (R); custom EM (Python); external:
  SHAPEIT, Beagle, Eagle for production.

## Assumptions & caveats

- **HWE and random mating** in the population.
- **No genotyping errors** — bad calls create phantom haplotypes.
- **EM local optima** at strong LD; multiple starts recommended.
- **Population reference panel** — production phasing uses matched-
  ancestry reference (1000 Genomes, TOPMed) for accuracy.

## Related in this repo

- `linkage-disequilibrium` — D and D' need phase; r² can use
  dosages.
- `gwas`, `polygenic-risk-scores` — downstream users of imputed
  dosages.

## Run

```
python techniques/haplotype-phasing/python/haplotype_phasing.py
Rscript techniques/haplotype-phasing/r/haplotype_phasing.R
```

**Refs:** Excoffier, L. & Slatkin, M. "Maximum-likelihood estimation of molecular haplotype frequencies in a diploid population." *Molecular Biology and Evolution*, 1995; Stephens, M., Smith, N.J., & Donnelly, P. "A new statistical method for haplotype reconstruction from population data." *American Journal of Human Genetics*, 2001.

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
