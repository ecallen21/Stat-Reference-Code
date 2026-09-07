# Combining p-values -- Fisher / Stouffer / Cauchy (Reference §22.16)

Fisher (1932); Stouffer et al. (1949); Liu & Xie (2020). Aggregate
`k` independent p-values into one overall p-value.

## Methods

| Method | Statistic | Reference distribution |
|---|---|---|
| **Fisher** | `X_F = −2 Σ log(p_i)` | `χ²_{2k}` |
| **Stouffer** | `Z = Σ w_i Φ⁻¹(1 − p_i) / √(Σ w_i²)` | `N(0, 1)` |
| **Cauchy (CCT)** | `T = Σ w_i tan((0.5 − p_i)π)` | Cauchy — robust to dependence |
| **Brown** | Fisher with correction | correlated `p_i` |
| **Tippett** | `min p_i` | Beta(1, k) |

## Files

- `python/fisher_combine_pvalues.py` — Fisher, Stouffer (weighted-Z),
  Cauchy CCT from scratch across 4 scenarios (null / signal / one
  strong + null / 10 mildly significant). Demo highlights: Fisher
  correctly rejects at 10 mildly-significant p-values (p_combined =
  2.5e-7); Cauchy is more conservative but stays valid under
  dependence.
- `r/fisher_combine_pvalues.R` — `metap::sumlog / sumz / allmetap`,
  `poolr`, `ACAT` (R); `scipy.stats.combine_pvalues`,
  `statsmodels.stats.combine_stats` (Python).

## When to use

- **Meta-analysis with p-values only** — study-level effect sizes not
  available.
- **Combining gene-set / SNP tests** — GWAS uses ACAT / Cauchy for
  dependent p-values.
- **Multi-endpoint clinical trials** — combining independent
  co-primary endpoint tests.

## When NOT to use

- **Effect sizes available** — proper meta-analysis (inverse-variance
  weighting) is more informative.
- **Dependent p-values** — Fisher over-rejects; use Brown, Cauchy,
  or empirical-null (Efron 2004).
- **Different directions of effect** — sum-of-log-p is sensitive to
  BOTH tails; use Stouffer with signed z's if direction matters.

## Assumptions & caveats

- **Independence** — Fisher's null distribution assumes independent
  p-values. Small violations bias mildly; strong correlation
  necessitates Brown / Cauchy.
- **Continuity** — p_i should be continuous under `H_0`; discrete
  p (e.g. permutation with few reps) inflates type I.
- **Uniform under `H_0`** — invalid if p-values are already
  conservative (e.g. exact tests on tiny cells).
- **Choice of weights (Stouffer)** — typically `√n_i` or inverse
  variance; equal weights easy but suboptimal.

## Related in this repo

- `meta-analysis`, `meta-regression` — effect-size-based analogues.
- `multiple-testing-corrections`, `multiple-metrics-fdr` — different
  problem: FDR / FWER control across tests, not aggregation into one
  p.
- `sequential-analysis`, `always-valid-inference` — testing at
  multiple looks in time.

## Run

```
python techniques/fisher-combine-pvalues/python/fisher_combine_pvalues.py
Rscript techniques/fisher-combine-pvalues/r/fisher_combine_pvalues.R
```

**Refs:** Fisher, R.A. *Statistical Methods for Research Workers*, 4th ed., Oliver & Boyd, 1932; Stouffer, S.A. et al. *The American Soldier*, Princeton, 1949; Liu, Y. & Xie, J. "Cauchy combination test: a powerful test with analytic p-value calculation under arbitrary dependency." *JASA*, 115(529): 393-402, 2020.

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
