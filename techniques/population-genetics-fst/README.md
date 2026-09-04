# Population-Genetics F_ST (Reference §40.20)

Weir & Cockerham (1984). **F_ST** measures allele-frequency
differentiation between subpopulations relative to the total:

```
F_ST = (H_T − H_S) / H_T
```

where `H_S` is the mean within-population heterozygosity and `H_T`
is total heterozygosity from the pooled allele frequency.

The Weir-Cockerham unbiased estimator handles finite sample size and
unequal subpopulation sizes.

## Interpretation (Wright 1978)

| F_ST | Level |
|---|---|
| < 0.05 | little differentiation |
| 0.05 – 0.15 | moderate |
| 0.15 – 0.25 | great |
| > 0.25 | very great |

## When to use

- **Population structure** description across geographic /
  ancestral groups.
- **Admixture / ancestry** inference (paired with STRUCTURE /
  ADMIXTURE).
- **Selection scans** — outlier F_ST loci suggest positive
  selection.

## When NOT to use

- **Single-locus questions** — F_ST is inherently multi-locus;
  per-locus estimates are noisy.
- **Very small subpopulations** — Weir-Cockerham correction helps
  but does not fix tiny `n_i`.

## Files

- `python/population_genetics_fst.py` — Weir-Cockerham per-locus
  F_ST + multi-locus average. Demo (K=3 populations × 100 samples
  × 40 loci): **panmixia F_ST ≈ 0.001**; **differentiated
  (allele freqs 0.2 / 0.5 / 0.7) F_ST ≈ 0.23** — "great" per
  Wright.
- `r/population_genetics_fst.R` — `hierfstat::pairwise.WCfst`,
  `pegas::Fst`, `adegenet::dapc`, `LEA::snmf` (R);
  `scikit-allel::weir_cockerham_fst` / `hudson_fst` (Python).

## Assumptions & caveats

- **Random mating within each subpopulation** — inbreeding shows up
  as F_IS and combines with F_ST to give F_IT.
- **Neutral loci assumption** — under-selection loci inflate F_ST.
- **Sample-size bias** — the WC estimator handles this; naive
  `(H_T − H_S) / H_T` overestimates F_ST at small `n_i`.
- **Confidence intervals via block-bootstrap** over loci
  (`hierfstat::boot.ppfst`).

## Related in this repo

- `gwas` — needs population-structure correction (PCs, LMM) when
  F_ST is high.
- `linkage-disequilibrium`, `hardy-weinberg`,
  `haplotype-phasing` — companion population-genetics QC.

## Run

```
python techniques/population-genetics-fst/python/population_genetics_fst.py
Rscript techniques/population-genetics-fst/r/population_genetics_fst.R
```

**Refs:** Weir, B.S. & Cockerham, C.C. "Estimating F-statistics for the analysis of population structure." *Evolution*, 1984; Pritchard, J.K., Stephens, M., & Donnelly, P. "Inference of population structure using multilocus genotype data." *Genetics*, 2000.

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
