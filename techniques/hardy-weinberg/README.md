# Hardy-Weinberg Equilibrium Testing (Reference §40.25)

Wigginton, Cutler & Abecasis (2005). Under **HWE** and random
mating, genotype frequencies at a bi-allelic locus are
`(q², 2pq, p²)` where `p = 1 − q` is the minor allele frequency.
Deviations are used as a GWAS quality-control filter — SNPs with
HWE p < 10⁻⁶ in controls typically flag genotyping errors.

## Two tests

- **χ² test** (1 df, approx.):
  `χ² = Σ (Obs − Exp)² / Exp`.
- **Exact test** (Wigginton 2005): enumerate all `n_Aa` configurations
  with the same allele totals and sum probabilities of configurations
  at most as likely as the observed. Preferred for rare variants
  and small n.

## When to use

- **GWAS quality control** on control cohort SNPs.
- **Population-genetics** deviation checks (founder effect,
  inbreeding, selection).

## When NOT to use

- **Cases in a case-control study** — under strong association a
  disease locus IS supposed to deviate; test in controls only.
- **Small MAF + small n** with the χ² approximation — the exact
  test is more reliable.

## Files

- `python/hardy_weinberg.py` — χ² test + Wigginton exact
  enumeration. Demo: HWE genotypes `(49, 42, 9)` give both p = 1.0;
  excess-heterozygote genotypes `(30, 60, 10)` give **χ² p =
  0.012**, **exact p = 0.021** — a possible genotyping-error flag.
- `r/hardy_weinberg.R` — `HardyWeinberg::HWExact` +
  `HWExactStats`, `genetics::HWE.test`, `pegas::hw.test` (R);
  `scikit-allel::hardy_weinberg_test`, `hail::hl.hardy_weinberg_test`
  + custom (Python).

## Assumptions & caveats

- **Random mating** — subpopulation structure creates HWE-like
  deviations without genotyping error (Wahlund effect).
- **No selection at the locus** — recent selection distorts
  genotype ratios.
- **Genotype calling quality** — deviation from HWE in controls is
  the standard signal of miscalling.

## Related in this repo

- `gwas` — the pipeline that uses HWE for QC.
- `linkage-disequilibrium`, `population-genetics-fst` — companion
  population-genetics QC.

## Run

```
python techniques/hardy-weinberg/python/hardy_weinberg.py
Rscript techniques/hardy-weinberg/r/hardy_weinberg.R
```

**Refs:** Wigginton, J.E., Cutler, D.J., & Abecasis, G.R. "A note on exact tests of Hardy-Weinberg equilibrium." *American Journal of Human Genetics*, 2005; Graffelman, J. & Moreno, V. "The mid p-value in exact tests for Hardy-Weinberg equilibrium." *Statistical Applications in Genetics and Molecular Biology*, 2013.

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
