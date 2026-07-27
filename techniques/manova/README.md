# MANOVA — Multivariate Analysis of Variance (Reference §9.2)

Extends ANOVA from a scalar response to a `p`-dimensional response vector. Given K groups and n_k p-dim observations per group, tests whether the group **mean vectors** are equal.

## The decomposition

```
T   =   H   +   E
(total)  (Hypothesis, between-groups SSCP)  (Error, within-groups SSCP)
```

Under `H₀: μ₁ = μ₂ = ... = μ_K`, the eigenvalues `λ_1..λ_s` of `E⁻¹ H` are small. Four common test statistics turn those eigenvalues into a scalar with an F-approximation:

| Statistic | Formula | Character |
|---|---|---|
| **Wilks' λ** | `∏ 1/(1 + λ_i)` | Likelihood ratio; most reported |
| **Pillai's trace** | `Σ λ_i/(1 + λ_i)` | Most robust to unequal covariances |
| **Hotelling–Lawley** | `Σ λ_i` | Generalization of Hotelling T² |
| **Roy's largest root** | `max_i λ_i` | Most powerful when only one dimension matters |

Wilks is the default in most software; Pillai when covariance equality is doubtful.

## Assumptions

- Multivariate normality of the residuals in each group.
- **Equal covariance matrices** across groups (Wilks/Hotelling/Roy sensitive to this; Pillai more robust). Test with Box's M when implemented.
- Independent observations.
- MANCOVA: same design but with covariates partialled out first.

## Files

- `python/manova.py` — SSCP decomposition, `E⁻¹H` eigenvalues, all four statistics with their F-approximations. Wilks / Pillai / Roy match `statsmodels.multivariate.manova.MANOVA` to 12 dp; Hotelling–Lawley uses a different denominator-df variant (both legitimate; produces same-order-of-magnitude p).
- `r/manova.R` — from-scratch + base `stats::manova` with all four test options.

## Run

```
python techniques/manova/python/manova.py
Rscript techniques/manova/r/manova.R
```

**Refs:** Rao, C.R. *Linear Statistical Inference and Its Applications*, 2nd ed., Wiley, 1973 (Ch. 8); Rencher, A.C. *Methods of Multivariate Analysis*, 2nd ed., Wiley, 2002 (Ch. 6); Johnson, R.A. & Wichern, D.W. *Applied Multivariate Statistical Analysis*, 6th ed., Pearson, 2007.

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
