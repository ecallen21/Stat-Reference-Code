# Log-Linear Models for Contingency Tables (Reference §8.1, §8.14)

A multi-way table of counts can be modeled as a **Poisson GLM with a log link** whose predictors are indicator variables for the classifying factors and their interactions. This gives a unified framework for testing conditional-independence structures and, for square agreement tables, decomposing agreement into chance, marginal bias, and symmetric-error components.

## The core idea

For a K-way table of counts `n_ijk...`, fit
```
log(μ_ijk...)  =  β₀  +  main_effects  +  chosen_interactions
```
via Poisson MLE (IRLS). The **residual deviance** `G² = 2 Σ n · log(n / μ)` is χ² with `(cells − parameters)` df under the null that the fitted model is adequate. Nested models are compared by ΔG² ~ χ²_{Δdf}.

**Hierarchical model space** for a 3-way (A, B, C) table:

| Notation | Meaning |
|---|---|
| `[A][B][C]` | mutual independence |
| `[AB][C]` | A×B associated, both indep of C |
| `[AB][AC]` | B and C conditionally indep given A |
| `[AB][AC][BC]` | all 2-way present, no 3-way (`ABC`) |
| `[ABC]` | saturated (`df = 0`) |

## Agreement models on a K×K square table

Same categories on rows and columns (e.g. rater 1 × rater 2). Extra structural models beyond independence:

| Model | Design term | What it captures |
|---|---|---|
| Independence | `R + C` | chance agreement only |
| **Quasi-independence** | `R + C + Diag` | independence + a bump on the diagonal |
| **Quasi-symmetry** | `R + C + Pair` | main effects + symmetric off-diagonal structure |
| Symmetry | `Pair` only, no separate main effects | strict symmetry `p_ij = p_ji` |

**LR test of quasi-symmetry vs. quasi-independence** decomposes: after allowing chance and diagonal excess (QI), does the *off-diagonal disagreement pattern* need symmetry? If so, disagreement is structured, not random.

## Files

- `python/log_linear_models.py` — IRLS Poisson fit, automatic design-matrix builder for any hierarchical model on an arbitrary K-way grid; deviance and LR nested-model comparisons; agreement-model suite. Deviances match `statsmodels.api.GLM(family=Poisson)` to 12 decimals.
- `r/log_linear_models.R` — from-scratch via `stats::glm(family=poisson)`; cross-checks `MASS::loglm` with Wilkinson–Rogers formulas.

## Assumptions

- Cells are Poisson (or equivalently multinomial conditional on the grand total).
- All cells fit share the sampling design that generated the counts.
- G² approximation is chi-square when expected counts ≥ 5 in most cells (Cochran's rule).

## Run

```
python techniques/log-linear-models/python/log_linear_models.py
Rscript techniques/log-linear-models/r/log_linear_models.R
```

**Refs:** Bishop, Y.M.M., Fienberg, S.E. & Holland, P.W. *Discrete Multivariate Analysis*, MIT Press, 1975; Agresti, A. *Categorical Data Analysis*, 3rd ed., Wiley, 2013 (Ch. 9–10); Fienberg, S.E. *The Analysis of Cross-Classified Categorical Data*, 2nd ed., Springer, 2007.

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
