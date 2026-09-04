# Rank-Based Inverse Normal Transformation (Reference §41.3)

Blom (1958), Beasley-Erickson-Allison (2009). Rank observations,
then apply the inverse standard-normal CDF. **Forces exact
normality** by construction; standard preprocessing for GWAS
quantitative traits and biomarker analysis.

## Variants

| Method | Quantile |
|---|---|
| Blom | `(r − 3/8) / (n + 1/4)` |
| Tukey | `(r − 1/3) / (n + 1/3)` |
| Van der Waerden | `r / (n + 1)` |
| Rankit (Bliss) | `(r − 0.5) / n` |

`r` = midrank of observation.

## When to use

- **GWAS phenotype** preprocessing — remove skew before per-SNP
  regression.
- **Biomarker discovery** — heavy-tailed lab values.
- **Downstream requires normality of residuals**.

## When NOT to use

- **Effect-size interpretability** matters — INT scales become
  z-units, not the original units.
- **Small n** — extreme ranks are unstable; the two most extreme
  points always map to fixed z-values.

## Files

- `python/inverse_normal_transformation.py` — four variants + skew,
  kurtosis, Shapiro-Wilk on transformed data. Demo (n=500,
  exponential y with skew 2.09): all four INT variants give
  **skew ≈ 0, Shapiro p ≈ 1.0**.
- `r/inverse_normal_transformation.R` — `RNOmni::RankNorm`,
  `bestNormalize::orderNorm` (R); `scipy.stats.rankdata` +
  `norm.ppf`, `sklearn.QuantileTransformer` (Python).

## Assumptions & caveats

- **Ties** — use midranks to avoid arbitrary tie-breaking.
- **Ignores metric** — INT discards the numeric spacing between
  values; conclusions on the transformed scale may distort effect
  magnitudes.
- **Post-INT effect sizes** are in standardised units; be explicit
  in reporting.
- **Not always beneficial** — Beasley et al. show INT is often
  routinely applied without checking whether the downstream method
  actually benefits.

## Related in this repo

- `box-cox-transformation`, `yeo-johnson-transformation` —
  parametric alternatives.
- `standardization-scaling` — post-transform standardisation.

## Run

```
python techniques/inverse-normal-transformation/python/inverse_normal_transformation.py
Rscript techniques/inverse-normal-transformation/r/inverse_normal_transformation.R
```

**Refs:** Beasley, T.M., Erickson, S., & Allison, D.B. "Rank-based inverse normal transformations are increasingly used, but are they merited?" *Behavior Genetics*, 2009; Blom, G. *Statistical Estimates and Transformed Beta-Variables*, Wiley, 1958.

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
