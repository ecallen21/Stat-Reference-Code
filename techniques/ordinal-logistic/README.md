# Ordinal Logistic Regression — Proportional Odds (Reference §7.2)

For an ordinal outcome `Y ∈ {1, 2, …, K}`, model the **K−1 cumulative logits** with a **shared** slope vector:

`logit P(Y ≤ k | x) = αₖ − x'β`,   `k = 1, …, K−1`

- `α₁ < α₂ < … < α_{K−1}` are category-specific intercepts (cutpoints).
- `β` is **the same at every cut** — this is the **proportional odds** assumption.
- `exp(βⱼ)` is the OR for moving up one category for a unit increase in `xⱼ`.

The proportional-odds form means the K−1 separate binary logistic regressions ("Y ≤ 1 vs > 1", "Y ≤ 2 vs > 2", …) should share the same slopes — only the intercepts differ.

## Fitting

Direct MLE via BFGS on a reparameterization that enforces `αₖ` ordering:
`αₖ = α₁ + Σ_{j<k} exp(δⱼ)` so the optimization is unconstrained in `(α₁, δ, β)`.

`MASS::polr` in R and `statsmodels.miscmodels.ordinal_model.OrderedModel` in Python use similar machinery; the implementation here matches statsmodels' parameters exactly on the demo.

## The proportional-odds assumption — and how to check it

The shared-slope assumption is restrictive. Two ways to test it:

1. **Brant test** (recommended; `brant::brant()` in R). Per-coefficient and overall test of proportional odds.
2. **Fit binary logistic at each cutpoint** and inspect whether the slopes look similar; informal but visual.

If the assumption fails for one or two predictors, use the **partial proportional odds model** (§7.3, deferred): relax the constraint for those predictors only. If it fails for many, switch to **multinomial logistic** (`techniques/multinomial-logistic`) — at the cost of giving up the ordering structure and gaining `(K−1)·p` parameters instead of `(K−1) + p`.

## Files
- `python/ordinal_logistic.py` — from-scratch BFGS on the cutpoint-reparameterized log-likelihood; matches `statsmodels.miscmodels.ordinal_model.OrderedModel`.
- `r/ordinal_logistic.R` — from-scratch via `optim` + `MASS::polr` cross-check.
- PySpark: N/A — Spark MLlib does not ship ordinal logistic. For massive ordinal outcomes, either fit K−1 binary logistic models (one per cutpoint) and check the slopes for compatibility, or sample down and use the Python version.

## Run
```
python techniques/ordinal-logistic/python/ordinal_logistic.py
Rscript techniques/ordinal-logistic/r/ordinal_logistic.R
```

**Refs:** McCullagh, "Regression Models for Ordinal Data," *JRSS-B* 42(2), 109–142, 1980; Agresti, *Analysis of Ordinal Categorical Data*, 2nd ed., Wiley, 2010.

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
