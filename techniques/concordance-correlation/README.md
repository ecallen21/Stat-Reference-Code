# Lin's Concordance Correlation Coefficient (Reference §4.9)

Measures **agreement** between two paired continuous measurements — not just linear association. Two devices whose readings are perfectly linearly related but **biased** (e.g. `y = x + 5`) have Pearson `r = 1` but Lin's CCC well below 1, because Pearson only measures the strength of the line, not whether it's the 45° identity line.

## Formula

`ρ_c = 2 · cov(x, y) / (var(x) + var(y) + (x̄ − ȳ)²)`

Equivalently — and this is the clean part — `ρ_c = r · C_b`:
- `r` = Pearson correlation = **precision** (how tight the points are around *some* line).
- `C_b = 2 / (v + 1/v + u²)` = **accuracy** (how close that line is to `y = x`),
  with `v = SD(x)/SD(y)` (scale shift) and `u = (x̄ − ȳ)/√(SD(x)·SD(y))` (location shift).

So `CCC = precision × accuracy`. `CCC = 1` iff `xᵢ = yᵢ` for every `i` — perfect agreement, not just perfect correlation.

## When to use

Method-comparison studies in clinical chemistry, imaging, sports science, sensor validation — any situation where you want to know if two measurements **agree**, not just whether they're linearly related.

For categorical agreement use Cohen's κ (Ch. 21 in the reference doc); for the variance-decomposition view (multiple raters per subject) see `techniques/intraclass-correlation` — `ICC(A, 1)` and Lin's CCC are very closely related for two raters.

## Confidence interval

Fisher z on `ρ_c`: `z = atanh(ρ_c)`, `SE = 1/√(n − 3)`, back-transform. Same form as Pearson's CI.

## Files
- `python/concordance_correlation.py` — from-scratch implementation with the precision/accuracy decomposition; demos three contrasts: bias-only (Pearson high, CCC low), perfect identity (both = 1), and scale shift `y = 2x + noise` (Pearson high, CCC near zero).
- `r/concordance_correlation.R` — from-scratch + `DescTools::CCC` (also `epiR::epi.ccc`).
- PySpark: N/A — closed form once the means / variances / covariance are aggregated; reuse `techniques/pearson-correlation/pyspark/` for the building blocks.

## Run
```
python techniques/concordance-correlation/python/concordance_correlation.py
Rscript techniques/concordance-correlation/r/concordance_correlation.R
```

**Refs:** Lin, "A Concordance Correlation Coefficient to Evaluate Reproducibility," *Biometrics* 45, 255–268, 1989; Lin, "Total Deviation Index for Measuring Individual Agreement With Applications in Laboratory Performance and Bioequivalence," *Statistics in Medicine* 19, 255–270, 2000.

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
