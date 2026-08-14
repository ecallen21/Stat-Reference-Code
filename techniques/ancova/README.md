# Analysis of Covariance (Reference §6.16)

Combines ANOVA (categorical predictor) with linear regression on a **continuous covariate** — typically a baseline value. Adjusts group means for the covariate to reduce within-group variance and increase power.

## Model

```
y_ij = μ + α_i + β · (x_ij − x̄) + ε_ij
```

- `α_i` — group effect (i-th treatment)
- `β` — common slope on covariate `x`
- **Parallel-slopes assumption** — the same `β` applies across groups.

## Two key tests

1. **Group effect**: F-test that `α_i` are all zero (adjusting for `x`).
2. **Parallel-slopes test**: fit `y ~ group + x + group:x` and F-test the interaction. If significant, ANCOVA is invalid — report Johnson-Neyman regions of significance instead.

## Adjusted means (LS means)

```
μ̂_i^adj = ȳ_i + β̂ · (x̄ − x̄_i)
```

Report these alongside raw group means; the adjusted means answer "what would the group mean be if all groups had covariate `x̄`?"

## Files

- `python/ancova.py` — full-vs-restricted F-test for group effect + full-vs-interaction F-test for parallel slopes + adjusted means. Demo (3 groups, common slope 0.5, group effects 1/2/3): F_group = 25.28 (p < 10⁻⁸); β̂_x = 0.525; parallel-slopes test p = 0.16 (fail to reject). Matches statsmodels exactly.
- `r/ancova.R` — base `aov(y ~ x + g)` + `emmeans::emmeans` for adjusted means.

## When to use

- **Randomized experiments with a baseline measurement** — ANCOVA is more powerful than change-score analysis (Vickers-Altman 2001).
- **Observational studies** where you want to adjust group means for one continuous confounder.
- **Educational / psychological studies** — pretest as covariate on posttest outcome.

## Assumptions & caveats

- **Parallel slopes** — always test.
- **Linear relationship** between `y` and `x` within each group.
- **Homogeneity of within-group residual variance**.
- **Covariate measured without error** — errors in `x` bias `β̂` (attenuation).
- **Randomized covariate independent of treatment** — in an RCT, this holds by design; in observational data, adjusting for a post-treatment variable can introduce collider bias.

## Related methods

- **Change-score analysis** — regress `(y_post − y_pre)` on group; equivalent to ANCOVA only if `β = 1`.
- **Multiple ANCOVA (MANCOVA)** — multivariate outcome.
- **Repeated-measures ANOVA** with baseline as first measurement — different assumptions.

## Run

```
python techniques/ancova/python/ancova.py
Rscript techniques/ancova/r/ancova.R
```

**Refs:** Fisher, R.A. *Statistical Methods for Research Workers*, Oliver & Boyd, 1925 (first published ANCOVA); Vickers, A.J. & Altman, D.G. "Analysing controlled trials with baseline and follow up measurements." *BMJ* 323(7321), 1123–1124, 2001.

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
