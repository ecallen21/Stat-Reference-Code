# Hotelling's T² Test (Reference §9.1, §9.28)

The multivariate generalization of the **t-test**. Given `p`-dimensional observations, test whether the mean vector equals a hypothesized value (one-sample) or whether two multivariate samples share a common mean vector (two-sample).

## Formulas

**One-sample** (compare mean vector to `μ₀`):

```
T²  =  n · (x̄ − μ₀)' S⁻¹ (x̄ − μ₀)
F   =  ((n − p) / (p(n − 1))) · T²        ~ F(p, n − p)
```

**Two-sample** (equal covariances, pooled S):

```
T²  =  (n₁·n₂ / (n₁ + n₂)) · (x̄₁ − x̄₂)' S_pool⁻¹ (x̄₁ − x̄₂)
F   =  ((n₁ + n₂ − p − 1) / (p(n₁ + n₂ − 2))) · T²   ~ F(p, n₁ + n₂ − p − 1)
```

`p = 1` reduces exactly to the ordinary t-test.

## Assumptions

- Multivariate normality of the observations (mild departures OK for large n).
- Two-sample: equal covariance matrices (test with `box-m` when implemented; if violated, use Welch-style modifications not implemented here).
- Independent observations.

## Files

- `python/hotellings_t2.py` — from-scratch one- and two-sample T²; F converts to a `statsmodels.multivariate.manova.MANOVA` Hotelling-Lawley cross-check (matches to 12 dp).
- `r/hotellings_t2.R` — from-scratch + `ICSNP::HotellingsT2`.

## Run

```
python techniques/hotellings-t2/python/hotellings_t2.py
Rscript techniques/hotellings-t2/r/hotellings_t2.R
```

**Refs:** Hotelling, H. "The generalization of Student's ratio." *Ann. Math. Stat.* 2(3), 360–378, 1931; Anderson, T.W. *An Introduction to Multivariate Statistical Analysis*, 3rd ed., Wiley, 2003 (Ch. 5); Rencher, A.C. *Methods of Multivariate Analysis*, 2nd ed., Wiley, 2002.

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
