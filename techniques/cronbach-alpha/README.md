# Cronbach's Alpha + McDonald's Omega (Reference §22.3)

Internal-consistency reliability of a `K`-item scale.

## Cronbach's α (1951)

```
α = (K / (K − 1)) · (1 − Σ_j Var(x_j) / Var(Σ_j x_j))
```

`α ∈ [0, 1]` (can be negative if items are anti-correlated). Rules of thumb:

- `α ≥ 0.7` — acceptable for research
- `α ≥ 0.8` — good for individual decisions
- `α ≥ 0.9` — high-stakes contexts (also worry about redundancy)

## Assumptions

- **Essential tau-equivalence** — all items measure the same trait with equal factor loadings (differ only in intercepts).
- **Uncorrelated errors**.
- Under these assumptions, α = reliability. When they fail (unequal loadings, correlated errors), α **underestimates** true reliability.

## McDonald's ω (relaxed)

Uses a **1-factor CFA** structure:

```
ω = (Σ λ_j)² / ((Σ λ_j)² + Σ θ_jj)          λ_j loadings, θ_jj residual variances
```

More appropriate when items have heterogeneous loadings.

## Files

- `python/cronbach_alpha.py` — from-scratch α + standardized α + mean inter-item r + α-if-item-deleted + eigendecomposition-based ω. Demo (K = 6, unequal loadings 0.6–0.8): α = 0.923, standardized α = 0.923, ω = 0.941 (slightly higher, reflecting unequal loadings).
- `r/cronbach_alpha.R` — `psych::alpha` and `psych::omega`.

## When to use

- **Any multi-item scale** — health questionnaires, personality inventories, cognitive tests.
- **Preliminary scale evaluation** before CFA (`cfa-confirmatory-factor`).
- **Item-deletion diagnostics** — remove items whose removal increases α.

## Cautions

- **Not a measure of unidimensionality** — a scale with two orthogonal factors can still have high α if within-factor correlations are moderate.
- **Depends on K** — long scales get high α mechanically.
- **Not sensitive to specific misfit** — a bad item can lurk unnoticed.
- **Compute on relevant sample** — norm-referenced α from an unrepresentative sample misleads.

## Related

- **Guttman's λ₆** — alternative reliability coefficient; more conservative.
- **KR-20** — special case of α for dichotomous items.
- **Split-half reliability + Spearman-Brown** (see `spearman-brown`).
- **Generalizability theory** (see `generalizability-theory`) — decomposes multiple facets.

## Run

```
python techniques/cronbach-alpha/python/cronbach_alpha.py
Rscript techniques/cronbach-alpha/r/cronbach_alpha.R
```

**Refs:** Cronbach, L.J. "Coefficient alpha and the internal structure of tests." *Psychometrika* 16(3), 297–334, 1951; McDonald, R.P. *Test Theory: A Unified Treatment*, Lawrence Erlbaum, 1999; Revelle, W. & Zinbarg, R.E. "Coefficients alpha, beta, omega, and the glb: comments on Sijtsma." *Psychometrika* 74(1), 145–154, 2009.

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
