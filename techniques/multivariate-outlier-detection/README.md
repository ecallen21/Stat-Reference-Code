# Multivariate Outlier Detection (Reference §9.6, §9.7)

Two complementary tools for finding outliers in **multivariate** data.

## Classical Mahalanobis distance

```
D²_i = (xᵢ − μ̂)ᵀ Σ̂⁻¹ (xᵢ − μ̂)
```

Under multivariate normality, `D²` is `χ²_p`. Flag `i` when `D²_i > χ²_{1−α, p}`.

**Problem — masking.** The sample mean and covariance are themselves dragged toward outliers. A cluster of outliers can inflate `Σ̂` so much that the outliers no longer look distant. Classical Mahalanobis alone can mask exactly what you're trying to detect.

## Robust Mahalanobis via MCD

Minimum Covariance Determinant (Rousseeuw 1985; Fast-MCD, Rousseeuw & Van Driessen 1999). Find the subsample of size `h ≈ 0.75 n` whose covariance has the **smallest determinant**; use its mean/covariance to compute robust Mahalanobis distances. Resists a contamination fraction of `(n − h) / n`.

```
minimize  |S_H|  over subsamples H of size h
μ̂_MCD, Σ̂_MCD = mean and covariance on the winning H
c = median(D²_MCD) / χ²_{0.5, p}     # consistency correction
```

## Files

- `python/mv_outlier_detection.py` — from-scratch classical Mahalanobis + Fast-MCD-flavored robust variant with C-step iteration and consistency correction. On synthetic data with a masked outlier cluster the robust version matches `sklearn.covariance.MinCovDet` exactly.
- `r/mv_outlier_detection.R` — `stats::mahalanobis` + `robustbase::covMcd`.

## When to use

- Any multivariate assumption check (LDA, MANOVA, regression influence).
- Sensor / instrument data with occasional gross errors.
- Fraud detection, quality control, novelty detection.

## Assumptions

- Multivariate normality is the reference distribution for the `χ²_p` cutoff. For heavy-tailed data, calibrate the cutoff empirically (e.g. via a data-driven quantile) instead of using `χ²`.
- `n > 2p` for a stable covariance; MCD needs `n > 2(p + 1)` at minimum and prefers `n ≥ 5p`.

## Run

```
python techniques/multivariate-outlier-detection/python/mv_outlier_detection.py
Rscript techniques/multivariate-outlier-detection/r/mv_outlier_detection.R
```

**Refs:** Mahalanobis, P.C. "On the generalized distance in statistics." *Proc. Natl. Inst. Sci. India* 2(1), 49–55, 1936; Rousseeuw, P.J. "Multivariate estimation with high breakdown point." *Math. Stat. Appl.* B, 283–297, 1985; Rousseeuw, P.J. & Van Driessen, K. "A fast algorithm for the Minimum Covariance Determinant estimator." *Technometrics* 41(3), 212–223, 1999.

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
