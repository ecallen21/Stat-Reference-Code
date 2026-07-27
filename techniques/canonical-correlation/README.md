# Canonical Correlation Analysis (Reference §9.29)

Given two sets of variables `X (n × p)` and `Y (n × q)` measured on the same subjects, CCA finds the linear combinations `U = X·a` and `V = Y·b` that are **maximally correlated**. The next pair `(u₂, v₂)` is the highest-correlation pair uncorrelated with the first, and so on for `min(p, q)` pairs.

Think of PCA on X × Y: instead of maximum variance within one set, we maximize *correlation across* two sets.

## Algorithm (schematic)

```
Rxx = X_c' X_c / (n − 1)
Ryy = Y_c' Y_c / (n − 1)
Rxy = X_c' Y_c / (n − 1)

M   = Rxx^(−½) · Rxy · Ryy⁻¹ · Rxy' · Rxx^(−½)
M   = A Λ A'      (eigendecomposition)

Canonical correlations   r_k  =  √λ_k
Canonical weights (X)    Wx   =  Rxx^(−½) · A
Canonical weights (Y)    Wy   =  Ryy⁻¹ · Rxy' · Wx / r
```

## Sequential test — Bartlett's chi-square on Wilks' Λ

For each k = 0, 1, ..., s − 1, test whether *any of* the remaining `s − k` canonical correlations are non-zero:

```
Λ_k  =  ∏_{j=k+1}^s  (1 − r_j²)
χ²   =  −(n − 1 − (p + q + 1)/2) · log Λ_k         df = (p − k)(q − k)
```

Chain gives you a data-driven answer for "how many canonical pairs are worth interpreting?"

## Files

- `python/canonical_correlation.py` — from-scratch CCA via generalized eigendecomposition + Bartlett sequential test; canonical correlations match `sklearn.cross_decomposition.CCA` to 9 dp.
- `r/canonical_correlation.R` — from-scratch + base `stats::cancor`.

## Assumptions

- Linear relationships between the two variable sets.
- No group of variables should be near-collinear within its set (otherwise the pseudoinverse steps in).
- Multivariate normality helps for Bartlett's test; CCA itself is descriptive.

## Run

```
python techniques/canonical-correlation/python/canonical_correlation.py
Rscript techniques/canonical-correlation/r/canonical_correlation.R
```

**Refs:** Hotelling, H. "Relations between two sets of variates." *Biometrika* 28(3/4), 321–377, 1936; Anderson, T.W. *An Introduction to Multivariate Statistical Analysis*, 3rd ed., Wiley, 2003 (Ch. 12); Härdle, W. & Simar, L. *Applied Multivariate Statistical Analysis*, 3rd ed., Springer, 2012.

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
