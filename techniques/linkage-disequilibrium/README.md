# Linkage Disequilibrium (Reference §40.26)

Slatkin (2008). Non-random association of alleles at different
loci. Three standard measures for two bi-allelic SNPs:

- **D** = `p_AB − p_A · p_B` — raw, scale-dependent.
- **D'** = `D / D_max` — normalised to `[−1, 1]`; `|D'| = 1` means
  complete LD.
- **r²** = `D² / (p_A · p_a · p_B · p_b)` — squared correlation;
  used for LD-pruning and LD score regression.

`D_max = min(p_A · p_b, p_a · p_B)` when `D > 0`, else
`min(p_A · p_B, p_a · p_b)`.

## When to use

- **GWAS QC / pruning** — keep only near-independent SNPs
  (`r² < 0.1`) for downstream analysis.
- **Fine-mapping** — LD structure determines the resolution.
- **Population history** — LD decay ~ 1 / (4 N_e c) reveals
  effective population size and time since admixture.

## When NOT to use

- **Cross-ancestry comparisons** — LD structure differs by
  ancestry; a LD estimate from one panel does not transfer.
- **Very small `n`** — sampling variance dominates D.

## Files

- `python/linkage_disequilibrium.py` — closed-form D / D' / r²
  from haplotype counts + pairwise r² matrix from dosage. Demo:
  perfect LD case → **r² = 1**; equilibrium → **r² = 0**; moderate
  → **D' = 0.60, r² = 0.36**; synthetic 20-SNP block with 10 %
  per-locus recombination rate decays r² from 1.00 (self) to ~0
  by SNP 2.
- `r/linkage_disequilibrium.R` — `genetics::LD`,
  `LDheatmap::LDheatmap`, `snpStats::ld`, `gaston::LD.plot` (R);
  `scikit-allel::rogers_huff_r`, `hail::hl.ld_matrix` (Python).

## Assumptions & caveats

- **Bi-allelic assumption** — extend to multi-allelic via
  pairwise decomposition.
- **Haplotype phase** must be known or inferred (see
  `haplotype-phasing`) for D / D' — r² can be computed from
  dosage alone.
- **Windowed r²** — a full genome-wide LD matrix is huge; use
  block or sliding-window computations (LDetect, PLINK).
- **Ancestral vs derived allele** choice affects D sign; standardise
  before comparing.

## Related in this repo

- `gwas`, `ld-score-regression`, `haplotype-phasing`,
  `hardy-weinberg`, `population-genetics-fst` — the population-
  genetics toolkit.
- `polygenic-risk-scores` — uses LD pruning.

## Run

```
python techniques/linkage-disequilibrium/python/linkage_disequilibrium.py
Rscript techniques/linkage-disequilibrium/r/linkage_disequilibrium.R
```

**Refs:** Slatkin, M. "Linkage disequilibrium: understanding the evolutionary past and mapping the medical future." *Nature Reviews Genetics*, 2008; Bulik-Sullivan, B.K. et al. "LD score regression distinguishes confounding from polygenicity in GWAS." *Nature Genetics*, 2015.

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
