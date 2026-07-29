# Box's M and Mauchly's Sphericity Tests (Reference §9.3, §12.2)

Two covariance-structure diagnostics used to check assumptions of downstream analyses.

## Box's M

Tests **equality of covariance matrices** across `K` groups — the assumption behind classical MANOVA and LDA.

```
M      = (n − K) log|S_pooled| − Σ_g (n_g − 1) log|S_g|
χ² ≈ (1 − c₁) · M          df = p(p+1)(K−1)/2
```

**Caveat.** Box's M is highly sensitive to non-normality and to sample size. With large `n`, even trivially different covariances yield tiny p-values. Interpret in the context of practical effect size; when it rejects, consider Pillai's trace (more robust) instead of Wilks' Λ, or a permutation MANOVA.

## Mauchly's sphericity

Tests whether the covariance of `p` **within-subject repeated measures** has the "spherical" form assumed by the univariate repeated-measures ANOVA F-test — equivalently, whether variances of all pairwise differences are equal.

```
C = M · S · Mᵀ            (M = orthonormal contrast, p−1 × p)
W = |C| / (tr(C) / (p−1))^(p−1)
χ² ≈ −(n − 1) · d · log W     df = p(p−1)/2 − 1
```

**When it rejects**, use either:
- Greenhouse–Geisser or Huynh–Feldt ε-corrected df (multiply the ANOVA df by ε).
- Multivariate approach (Pillai / Wilks) that makes no sphericity assumption.
- A mixed-model (`nlme::lme` / `lme4::lmer`) with an unstructured or AR(1) covariance.

## Files

- `python/box_m_mauchly.py` — from-scratch Box's M and Mauchly's W using SciPy's χ² CDF; also returns Greenhouse–Geisser and Huynh–Feldt ε for downstream ANOVA df correction.
- `r/box_m_mauchly.R` — wrappers around `heplots::boxM` and `stats::mauchly.test`.

## Assumptions

- **Box's M**: multivariate normality within each group. Highly sensitive to violation.
- **Mauchly's**: multivariate normality of the within-subject repeated measures; balanced design with `p ≥ 3` timepoints.

## Run

```
python techniques/box-m-mauchly/python/box_m_mauchly.py
Rscript techniques/box-m-mauchly/r/box_m_mauchly.R
```

**Refs:** Box, G.E.P. "A general distribution theory for a class of likelihood criteria." *Biometrika* 36(3/4), 317–346, 1949; Mauchly, J.W. "Significance test for sphericity of a normal n-variate distribution." *Ann. Math. Stat.* 11(2), 204–209, 1940; Greenhouse, S.W. & Geisser, S. "On methods in the analysis of profile data." *Psychometrika* 24(2), 95–112, 1959.

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
