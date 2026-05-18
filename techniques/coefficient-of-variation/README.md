# Coefficient of Variation (Reference §1.22, §1.33)

`CV = SD / mean` — a **unit-free** measure of relative variability. Lets you compare spread between variables on different scales or with different means (often reported as a percentage).

## Variants
| Variant | Formula | Use |
|---------|---------|-----|
| Sample CV | `s / x̄` (×100%) | General relative variability |
| Geometric CV | `√(exp(s_log²) − 1)`, `s_log = SD(log x)` | Log-normal data (assay concentrations, titers); ≈ CV for small CV; relates to `GSD − 1` |
| Within-subject CV (CV_w) | `√(mean of within-subject variances) / overall mean` | Assay precision, biological-variation studies (CVI) — needs replicate measurements per subject |

**Confidence interval** — McKay's chi-squared approximation (good for CV ≲ 0.33):
with `v = n−1`, `CV ∈ [ k/√((χ²_{1−α/2,v}/v − 1)k² + χ²_{1−α/2,v}/v), k/√((χ²_{α/2,v}/v − 1)k² + χ²_{α/2,v}/v) ]` where `k` is the sample CV.

## When NOT to use it
- Mean near zero (CV explodes) or data with mixed signs (meaningless).
- Interval scales with an arbitrary zero (temperature in °C).

Rules of thumb seen in practice: CGM glycemic-variability CV < 36% = "stable"; assay precision CV < 15% acceptable. These are field conventions, not universal laws.

## Files
- `python/coefficient_of_variation.py` — from-scratch sample / geometric / within-subject CV + McKay CI; compares against `scipy.stats.variation`
- `r/coefficient_of_variation.R` — from-scratch + `sd/mean`, `DescTools::CoefVar`; notes on `raster::cv`, `rstatix`
- `pyspark/coefficient_of_variation.py` — `stddev_samp/mean` aggregation; within-subject CV via per-subject `var_samp` then a global average

## Run
```
python techniques/coefficient-of-variation/python/coefficient_of_variation.py
Rscript techniques/coefficient-of-variation/r/coefficient_of_variation.R
python techniques/coefficient-of-variation/pyspark/coefficient_of_variation.py
```

**Refs:** Abdi, "Coefficient of Variation," in *Encyclopedia of Research Design*, SAGE, 2010; Reed, Lynn & Meade, "Use of Coefficient of Variation in Assessing Variability of Quantitative Assays," *Clin. Diagn. Lab. Immunol.* 9(6), 1235–1239, 2002.

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
