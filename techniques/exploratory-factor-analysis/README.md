# Exploratory Factor Analysis (Reference §9.4)

EFA models each observed variable `X_j` as a linear combination of `k` **unobserved common factors** plus a variable-specific "unique" component:

```
X_j  =  Λ_j1 F_1 + ... + Λ_jk F_k  +  U_j
```

so the correlation / covariance matrix decomposes as

```
Σ  =  Λ Λ'  +  Ψ
      ↑common↑   ↑diagonal uniquenesses↑
```

Unlike PCA (which is a pure orthogonal rotation of the observed variables), EFA posits latent factors and separates variance into what's *shared* (common) vs. *variable-specific* (unique).

## Extraction

- **Principal Axis Factoring (PAF)** — iterate on the correlation matrix with the diagonal replaced by current-iteration communalities. Simple, robust default; doesn't require distributional assumptions.
- ML extraction (available in libraries) gives a proper LR test of number of factors under multivariate normality; not implemented from scratch here.

## Rotation

Rotation doesn't change the fit but makes the loadings interpretable ("simple structure").

- **Varimax** (orthogonal): maximizes the variance of squared loadings within each column — each factor loads highly on a few variables.
- **Promax** (oblique): raise varimax loadings to power κ (typically 4) → Procrustes target → oblique rotation. Factors are allowed to correlate; the correlation matrix `Φ` (with unit diagonal) is returned alongside.

## Key outputs

| Quantity | Meaning |
|---|---|
| **Loadings `Λ`** | `p × k` matrix; correlation of each variable with each factor |
| **Communalities `h²_j`** | proportion of variable j's variance explained by the common factors (row sums of squared loadings; oblique: `Λ_j Φ Λ_j'`) |
| **Uniquenesses** | `1 − h²_j` |
| **Factor correlation `Φ`** (oblique only) | k × k with unit diagonal |

## Files

- `python/exploratory_factor_analysis.py` — PAF extraction; varimax and promax rotations from scratch. Loadings recover a synthetic 2-factor DGP cleanly; oblique Φ has proper unit diagonal.
- `r/exploratory_factor_analysis.R` — from-scratch + `psych::fa` when installed.

## Assumptions

- Continuous variables that are (approximately) multivariate normal.
- Number of factors `k` chosen a priori — inspect a scree plot or use `cluster-validation`-style criteria to pick.
- **Not** a statistical *test* of latent structure — for that, use CFA (Chapter 19 SEM).

## Run

```
python techniques/exploratory-factor-analysis/python/exploratory_factor_analysis.py
Rscript techniques/exploratory-factor-analysis/r/exploratory_factor_analysis.R
```

**Refs:** Fabrigar, L.R., Wegener, D.T., MacCallum, R.C. & Strahan, E.J. "Evaluating the use of exploratory factor analysis in psychological research." *Psych. Methods* 4(3), 272–299, 1999; Hendrickson, A.E. & White, P.O. "Promax: A quick method for rotation to oblique simple structure." *Br. J. Stat. Psych.* 17(1), 65–70, 1964; Kaiser, H.F. "The varimax criterion for analytic rotation in factor analysis." *Psychometrika* 23(3), 187–200, 1958.

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
