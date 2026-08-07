# Multiple-Testing Corrections (Reference §3.30, §4.24)

`m` p-values from a family of tests. Uncorrected `α = 0.05` gives an expected ~5% false-positive **per test**, but the probability of at least one false positive grows toward 1 with `m`. Two families of corrections:

## FWER — family-wise error rate

Probability of **any** false positive. Very conservative.

- **Bonferroni**: reject if `p_i ≤ α / m`. Simple; usually too strict.
- **Šidák**: `p_i ≤ 1 − (1 − α)^(1/m)`. Tighter than Bonferroni under independence.
- **Holm** (step-down): sort p-values; reject in ascending order while `p_(k) ≤ α / (m − k + 1)`. Uniformly more powerful than Bonferroni.
- **Hochberg** (step-up): similar; needs positive dependence.

## FDR — false discovery rate

Expected proportion of rejections that are false. Less conservative; standard for large-scale testing (genomics, imaging, A/B testing).

- **Benjamini-Hochberg (BH, 1995)**: reject the first `K` where `p_(K) ≤ K/m · α`. Valid under independence or PRDS positive dependence.
- **Benjamini-Yekutieli (BY, 2001)**: BH with a `log(m)` inflation factor — valid under arbitrary dependence.
- **Storey q-value (2002)**: adaptive BH that estimates the fraction `π₀` of true nulls and uses `(π₀ m / K) α` instead. More power when `π₀ < 1`.

## Adjusted p-values

`p_adj_i` = smallest `α` at which hypothesis `i` is rejected. Report these alongside raw p-values.

## Files

- `python/multiple_testing_corrections.py` — from-scratch Bonferroni, Šidák, Holm, Hochberg, BH, BY, and Storey q. Demo (50 tests, 10 true effects): Bonferroni / Holm / Hochberg / Šidák each reject 2 (2 TP / 0 FP); BH rejects 9 (8 TP / 1 FP); Storey-q with π̂₀ = 0.96 also 9. Matches `statsmodels.stats.multitest.multipletests`.
- `r/multiple_testing_corrections.R` — base R `p.adjust(method = c("bonferroni", "holm", "hochberg", "BH", "BY"))` + `qvalue::qvalue` for Storey q.

## Choosing the right control

- **Regulatory / single decision**: FWER (Bonferroni or Holm).
- **Screening (many tests, tolerate some FPs)**: FDR (BH). Genomics, GWAS, imaging use BH by default.
- **Under strong positive dependence** (spatial data): BY or permutation-based FWER.
- **Under weak dependence with many nulls**: Storey q for extra power.

## Assumptions & caveats

- **BH** assumes independence or PRDS (positive regression dependence on the null subset). Most real studies satisfy this.
- **BY** is safe under arbitrary dependence but conservative.
- **Storey q** needs enough p-values to estimate `π₀` reliably (typically `m ≥ 100`).
- **Report the correction method** always — "FDR-adjusted BH q < 0.05" is not the same claim as "raw p < 0.05".

## Run

```
python techniques/multiple-testing-corrections/python/multiple_testing_corrections.py
Rscript techniques/multiple-testing-corrections/r/multiple_testing_corrections.R
```

**Refs:** Holm, S. "A simple sequentially rejective multiple test procedure." *Scand. J. Stat.* 6(2), 65–70, 1979; Benjamini, Y. & Hochberg, Y. "Controlling the false discovery rate: a practical and powerful approach to multiple testing." *J. R. Stat. Soc. B* 57(1), 289–300, 1995; Storey, J.D. "A direct approach to false discovery rates." *J. R. Stat. Soc. B* 64(3), 479–498, 2002.

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
