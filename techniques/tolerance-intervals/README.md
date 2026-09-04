# Tolerance Intervals (Reference §38.16)

Krishnamoorthy & Mathew (2009). A `(P, 1−α)` **tolerance interval**
covers at least proportion `P` of the population with confidence
`1 − α`. Distinct from confidence and prediction intervals:

| Interval | About |
|---|---|
| Confidence | a parameter (e.g. `μ`) |
| Prediction | a single future observation |
| **Tolerance** | a **proportion of the population** |

## Normal two-sided (Howe 1969)

```
[x̄ − k₂ s,  x̄ + k₂ s]
k₂ ≈ z_{(1+P)/2} · √( (n−1)(1 + 1/n) / χ²_{α, n−1} )
```

## Nonparametric (Wilks 1941)

Use order statistics `X_(r), X_(n−r+1)`. The Beta-Binomial identity
gives the coverage-confidence: pick the largest `r` such that
`P{coverage ≥ P} ≥ 1 − α`. `r = 1` uses min/max; larger `r` gives
narrower intervals but requires larger `n`.

## When to use

- **Quality control** — "95 % of parts fall within these
  dimensions, with 95 % confidence."
- **Reference ranges** — normal ranges for a lab assay covering
  most of the healthy population.
- **Compliance / regulation** — "no more than 5 % out of spec."

## When NOT to use

- **You want inference on the mean** — that's a confidence
  interval.
- **A single future prediction** — that's a prediction interval.
- **Very small `n`** — even nonparametric TIs may fail to exist at
  the requested `(P, 1−α)`.

## Files

- `python/tolerance_intervals.py` — Howe 1969 normal TI + Wilks
  1941 nonparametric TI. Demo (n=60, `N(100, 15)` data): normal
  `(P=0.95, 95 %)` TI = **[69.5, 132.8]**; nonparametric only
  supports `P=0.90` at this `n`; simulated coverage confidence
  **0.946 vs target 0.95**.
- `r/tolerance_intervals.R` — `tolerance` (normtol.int,
  nptol.int, exttol.int) (R); custom (Python).

## Assumptions & caveats

- **Normal TI assumes normality** — non-normal data need
  distribution-specific (`exttol.int` for exponential) or
  nonparametric intervals.
- **Nonparametric `r` grows** — reaching high `P` needs
  large `n` (`P = 0.99, α = 0.05` needs `n ≥ 473` for `r = 1`).
- **Two-sided vs one-sided** — one-sided TIs need one-sided
  formulae (asymmetric).
- **Sample independence** — assumed by both forms; time-series or
  clustered data need adjustment.

## Related in this repo

- `confidence-intervals` (implicit in many techniques) — parameter
  intervals.
- `conformal-prediction` — nonparametric prediction with finite-
  sample coverage guarantees.
- `extreme-value-theory` — for very extreme tail quantiles.

## Run

```
python techniques/tolerance-intervals/python/tolerance_intervals.py
Rscript techniques/tolerance-intervals/r/tolerance_intervals.R
```

**Refs:** Krishnamoorthy, K. & Mathew, T. *Statistical Tolerance Regions: Theory, Applications, and Computation*, Wiley, 2009; Howe, W.G. "Two-sided tolerance limits for normal populations." *JASA*, 1969; Wilks, S.S. "Determination of sample sizes for setting tolerance limits." *Annals Math Statist*, 1941; NIST/SEMATECH e-Handbook of Statistical Methods, Sec 7.2.6.

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
