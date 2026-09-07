# Meta-Regression (Reference §22.5)

Thompson & Sharp (1999), Viechtbauer (2010). Extension of random-
effects meta-analysis with a per-study covariate (moderator) to
explain between-study heterogeneity:

```
y_i = β_0 + β_1 · x_i + u_i + ε_i
u_i ~ N(0, τ²),   ε_i ~ N(0, σ_i²)
```

Estimated by REML (or DL / ML / SJ variants). Report
`R²_heterog = 1 − τ²_res / τ²_null` — proportion of between-study
variance explained by the moderator.

## When to use

- **Systematic reviews** with substantial heterogeneity worth
  explaining.
- **Dose-response, publication-year, quality-score** moderators.
- **Subgroup comparisons** as a formal alternative to arm-by-arm
  reporting.

## When NOT to use

- **Fewer than ~10 studies** — meta-regression is under-powered;
  no `τ²` decomposition is reliable.
- **Ecological fallacy** risk — study-level covariates cannot
  answer individual-level questions.

## Files

- `python/meta_regression.py` — REML for τ² via 1-D optimisation +
  GLS for β̂ + null-model R² decomposition. Demo (K=20 studies,
  true β=(0.5, −0.8), τ=0.10, per-study SE 0.05-0.20): recovered
  **intercept +0.57 (SE 0.07), slope −0.84 (SE 0.12), τ²=0.012**;
  **86.7 % of heterogeneity explained**.
- `r/meta_regression.R` — `metafor::rma`/`rma.mv`, `meta::metareg`,
  `brms` (R); `statsmodels`, `pymare` + custom (Python).

## Assumptions & caveats

- **Moderator quality** — a proxy that mis-classifies studies
  cannot explain heterogeneity.
- **Continuous vs categorical moderator** — categorical needs
  ≥ 3-4 studies per level.
- **Multiple moderators** — inflate multiple-testing risk; pre-
  specify or use permutation tests (`metafor::permutest`).
- **Random-effects assumption** — heterogeneous studies are drawn
  from a common distribution; misspecified when true effects vary
  systematically.

## Related in this repo

- `meta-analysis` (if present) — parent method.
- `trim-fill` — publication-bias adjustment.
- `hierarchical-models` — the underlying random-effects framework.

## Run

```
python techniques/meta-regression/python/meta_regression.py
Rscript techniques/meta-regression/r/meta_regression.R
```

**Refs:** Thompson, S.G. & Sharp, S.J. "Explaining heterogeneity in meta-analysis: a comparison of methods." *Statistics in Medicine*, 1999; Viechtbauer, W. "Conducting meta-analyses in R with the metafor package." *Journal of Statistical Software*, 2010.

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
