# Split-Plot Design (Reference §18.x extra)

Two-factor design with **restricted randomisation**: one factor (whole-plot,
`A`) is harder or more expensive to change than the other (subplot, `B`).
Whole-plots receive levels of `A`; each whole-plot is split into subplots that
receive levels of `B`.

## Two error strata

- **Whole-plot error (error a)**: variability between whole-plots within the same A level. Denominator for testing `A`.
- **Subplot error (error b)**: variability between subplots within the same whole-plot. Denominator for testing `B` and `A × B`.

## ANOVA table (balanced, a levels of A × r replicate whole-plots × b subplots)

| Source | df | Denominator | Notes |
|---|---|---|---|
| A (whole-plot) | `a − 1` | **MSE_WP** | fewer replicates → larger MS_WPE → less power |
| WP within A (error a) | `a (r − 1)` | — | whole-plot variance component |
| B (subplot) | `b − 1` | **MSE_SP** | more replicates → smaller MS_SP → more power |
| A × B | `(a − 1)(b − 1)` | **MSE_SP** | usually well-powered |
| Residual (error b) | `a (r − 1)(b − 1)` | — | subplot variance component |
| **Total** | `abr − 1` | | |

## The classic split-plot mistake

Analysing all `abr` subplots as though they were independent replicates
(a completely-randomised two-way ANOVA) uses one error term for both `A` and
`B`. This inflates the F-statistic for `A` because the whole-plot variance is
ignored — the demo below shows `F_A` jumping from 6.1 (correct) to 19.6
(naive) and `p` from 0.02 to <0.0001.

## When to use

- **Agricultural field trials** — irrigation (whole-plot) × fertiliser (subplot).
- **Industrial experiments** — oven temperature (whole-plot) × material (subplot).
- **Clinical / behavioural** — clinic-level intervention × patient-level covariate.
- **Any factor combination where randomisation is restricted** at one level.

## Files

- `python/split_plot_design.py` — from-scratch balanced split-plot ANOVA. Demo (a=3 A levels × r=4 whole-plot reps × b=4 subplots + true WP random effect SD = 1.2 and residual SD = 0.5): correctly gets F_A = 6.12 (p = 0.021) with error a, F_B = 27.9 (p < 0.001) with error b, no A × B; naive one-error ANOVA inflates F_A to 19.58 (p < 0.0001).
- `r/split_plot_design.R` — `aov(y ~ A * B + Error(WP / B))`, `lme4::lmer(y ~ A * B + (1 | WP))`, `afex::aov_car`, `agricolae::sp.plot`.

## Assumptions & caveats

- **Balanced design** for the ANOVA table above; unbalanced designs need mixed-model machinery (`lme4::lmer`).
- **Normality + homoscedasticity within strata** — same standard assumptions as one-way ANOVA.
- **Sphericity for the subplot factor** if B has > 2 levels and comes from a within-subject repeated-measures context — see `repeated-measures-anova` and Mauchly's test in `box-m-mauchly`.
- **Whole-plot degrees of freedom** are usually small — the test for `A` has low power; plan enough whole-plot replicates.
- **Strip-plot / criss-cross / split-split-plot** designs extend this to more restricted-randomisation layers with more error strata.

## Related in this repo

- `latin-square-design`, `fractional-factorial`, `response-surface` — other systematic designs.
- `repeated-measures-anova`, `linear-mixed-models` — the split-plot is a special case of a linear mixed model.
- `kenward-roger` — small-sample adjustment for tests derived from mixed-model fits.

## Run

```
python techniques/split-plot-design/python/split_plot_design.py
Rscript techniques/split-plot-design/r/split_plot_design.R
```

**Refs:** Yates, F. "Complex experiments." *JRSS Suppl.* 2, 181–247, 1935; Cochran, W.G. & Cox, G.M. *Experimental Designs*, 2nd ed., Wiley, 1957; Jones, B. & Nachtsheim, C.J. "Split-plot designs: what, why, and how." *J. Qual. Tech.* 41(4), 340–361, 2009.

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
