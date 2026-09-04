# Discretization / Binning (Reference §41.7)

Royston-Altman-Sauerbrei (2006), Fayyad-Irani (1993),
Harrell (2015 ch 2). Three families for turning a continuous
predictor into a categorical one:

- **Equal-width** — split range into `k` intervals of equal width.
- **Equal-frequency** — `k` intervals with equal counts (quantiles).
- **Entropy-based** (Fayyad-Irani MDL) — split points chosen to
  maximise information about the target.

## The Royston-Altman-Sauerbrei warning

**Dichotomising a continuous predictor is almost always a bad idea.**
It:
- discards information → power loss;
- forces a step function where a smooth relationship exists;
- creates spurious interactions;
- makes results depend on the cutoff choice, which is often
  data-driven.

Use restricted cubic splines (`Hmisc::rcs`) for a continuous
predictor unless clinical convention absolutely requires a cutoff.

## When to use

- **Data-mining exploration** where non-monotone effects need
  quick visualisation.
- **Reporting** where the audience only understands "high / normal /
  low".
- **Downstream model requires categorical input** (association rule
  mining).

## When NOT to use

- **Predictive modelling** — a spline or regression on the
  continuous variable dominates.
- **Small samples** — extra df from bins wastes power.

## Files

- `python/discretization_binning.py` — equal-width, equal-freq,
  Fayyad-Irani MDL entropy binning. Demo (n=500, exponential x,
  binary y = 1{x > 2.5}): entropy binner recovers **cut at 2.50**
  exactly. Discretisation cost: fit acc for logistic on
  **continuous 1.00**, **dichotomise-at-median 0.82**, **dichotomise
  at true 2.5 → 1.00**, **entropy bins 0.998**.
- `r/discretization_binning.R` — `arules::discretize`,
  `Hmisc::cut2`, `infotheo::discretize` (R);
  `sklearn.KBinsDiscretizer`, `pandas.cut`/`qcut`, `optbinning`
  (Python).

## Assumptions & caveats

- **Cutoff choice** — data-driven cutoffs (e.g., maximally-selected
  rank statistics) inflate p-values unless multiple-testing
  corrected.
- **Downstream df** — many bins waste df; few bins hide non-
  linearity.
- **Splines** are almost always the more honest alternative for
  predictive or inferential models.

## Related in this repo

- `shape-constrained-regression`, `additive-quantile-regression`
  — smooth alternatives to binning.
- `cutpoint-methods` (if present) — for when a threshold is
  clinically required.

## Run

```
python techniques/discretization-binning/python/discretization_binning.py
Rscript techniques/discretization-binning/r/discretization_binning.R
```

**Refs:** Royston, P., Altman, D.G., & Sauerbrei, W. "Dichotomizing continuous predictors in multiple regression: a bad idea." *Statistics in Medicine*, 2006; Fayyad, U. & Irani, K. "Multi-interval discretization of continuous-valued attributes." *IJCAI*, 1993; Harrell, F.E. *Regression Modeling Strategies*, 2nd ed., Springer, 2015 (ch 2).

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
