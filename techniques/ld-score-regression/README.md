# LD Score Regression (Reference §40.17)

Bulik-Sullivan et al. (2015). From GWAS summary statistics + an LD
reference panel, estimate **SNP heritability `h²_g`** and — critically
— separate polygenic signal from confounding inflation.

## Model

For SNP `j` with LD score `ℓ_j = Σ_k r²_jk`:

```
E[ χ²_j ] = (N · h²_g / M) · ℓ_j + N · a + 1
```

Regress `χ²` on `ℓ`:

- **Slope** ∝ heritability: `h²_g = slope · M / N`.
- **Intercept** = `1 + N · a` where `a` is confounding bias.
  - Intercept ≈ 1 → polygenic signal only.
  - Intercept ≫ 1 → population stratification / cryptic
    relatedness inflating tests.

`λ_GC` alone cannot distinguish these; LDSC intercept can.

## When to use

- **Summary-stat-only** heritability + genetic-correlation
  estimation (no individual-level data required).
- **Cross-trait comparison** — LDSC's bivariate variant gives
  genetic correlation `r_g`.
- **GWAS QC** — a high intercept flags confounding you need to fix.

## When NOT to use

- **Small `M`** or non-European LD reference — biased estimates.
- **Trans-ethnic transfer** — LD structure differs by ancestry.
- **Very low `h²`** — power to detect nonzero slope is weak.

## Files

- `python/ld_score_regression.py` — unweighted OLS reference
  implementation. Demo (M=5000 SNPs, N=20 000, true h²=0.30,
  confounding a=0.002): **estimated h² = 0.300**; **intercept
  estimate 41.9 vs truth 41.0** (large due to non-zero `a`).
- `r/ld_score_regression.R` — `GenomicSEM`, `bigsnpr::snp_ldsc` (R);
  `ldsc` (Broad reference), `hail::ld_score_regression`, custom
  (Python).

## Assumptions & caveats

- **LD reference panel matches** GWAS ancestry — mis-match biases
  both estimates.
- **HapMap3 SNPs** — the standard LDSC panel restricts to well-
  imputed common variants.
- **Weights** — full LDSC uses iterated GLS with 1/(1 + N·h²·ℓ/M)²
  weights; the reference here uses unit weights.
- **Bivariate LDSC** for genetic correlation between two traits is
  the more common downstream use.

## Related in this repo

- `gwas` — where the χ² come from.
- `linkage-disequilibrium` — how ℓ is computed.
- `mendelian-randomization` — related summary-stat framework.

## Run

```
python techniques/ld-score-regression/python/ld_score_regression.py
Rscript techniques/ld-score-regression/r/ld_score_regression.R
```

**Refs:** Bulik-Sullivan, B., Loh, P.-R., Finucane, H.K. et al. "LD score regression distinguishes confounding from polygenicity in genome-wide association studies." *Nature Genetics*, 2015; Yang, J., Benyamin, B., McEvoy, B.P. et al. "Common SNPs explain a large proportion of the heritability for human height." *Nature Genetics*, 2010.

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
