# Cohen's Kappa: Agreement Between Two Raters (Reference §8.4)

Chance-corrected agreement between two raters classifying `n` items into K nominal categories.

```
                p_o - p_e
κ  =  ─────────
                1 - p_e

p_o = Σ_i n_ii / n                (observed agreement rate)
p_e = Σ_i (row_i · col_i) / n²    (expected under independence)
```

`κ = 1` is perfect agreement; `κ = 0` is exactly chance; negative values indicate less agreement than chance (rare in practice).

**Landis–Koch benchmarks**: `<0.20` slight · `0.21–0.40` fair · `0.41–0.60` moderate · `0.61–0.80` substantial · `0.81–1.00` almost perfect.

## Statistics computed

| Statistic | Description |
|---|---|
| `κ` | Cohen's kappa |
| **ASE** (Fleiss, 1969) | asymptotic SE for a Wald z-test of `κ ≠ 0` |
| **95% Wald CI** | `κ ± 1.96 · ASE` |
| **PABAK** (Byrt et al., 1993) | prevalence-and-bias-adjusted κ: `(K · p_o − 1) / (K − 1)` — depends only on the diagonal sum |

**Why PABAK?** Kappa can drop toward zero even at 90%+ agreement when one category dominates (high *prevalence*), or when the two raters use categories at very different overall rates (high *bias*). PABAK is a sanity check.

## Files

- `python/cohens_kappa.py` — from-scratch κ, Fleiss ASE, PABAK; cross-checks `sklearn.metrics.cohen_kappa_score` and `statsmodels.stats.inter_rater.cohens_kappa`.
- `r/cohens_kappa.R` — from-scratch + `irr::kappa2` / `psych::cohen.kappa`.
- `pyspark/cohens_kappa.py` — `groupBy(rater1, rater2).count()` builds the K×K confusion matrix from a Spark DataFrame; scalar test on the driver.

## Assumptions

- Two raters, K nominal categories (for ordinal, use `weighted-kappa`; for ≥3 raters, use `fleiss-kappa`).
- Same categorization scheme; **independent** ratings across items.

## Run

```
python techniques/cohens-kappa/python/cohens_kappa.py
Rscript techniques/cohens-kappa/r/cohens_kappa.R
python techniques/cohens-kappa/pyspark/cohens_kappa.py
```

**Refs:** Cohen, J. "A coefficient of agreement for nominal scales." *Educational and Psychological Measurement* 20(1), 37–46, 1960; Fleiss, J.L., Cohen, J. & Everitt, B.S. "Large sample standard errors of kappa and weighted kappa." *Psych. Bull.* 72(5), 323–327, 1969; Byrt, T., Bishop, J. & Carlin, J.B. "Bias, prevalence and kappa." *J. Clin. Epidemiol.* 46(5), 423–429, 1993; Landis, J.R. & Koch, G.G. "The measurement of observer agreement for categorical data." *Biometrics* 33(1), 159–174, 1977.

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
