# Wald SPRT (Reference §47.346)

Wald (1945). Sequential Probability Ratio Test for two simple
hypotheses `H₀: θ = θ₀` vs `H₁: θ = θ₁`. Accumulate the
log-likelihood ratio:

```
LR_n = Σ log(f_{θ₁}(x_i) / f_{θ₀}(x_i))
```

Boundaries `A = log((1 − β)/α)` and `B = log(β/(1 − α))`:

```
LR_n ≥ A   → reject H₀
LR_n ≤ B   → reject H₁
else       → keep sampling
```

Wald-Wolfowitz optimality: SPRT MINIMISES expected sample size
among all tests with the same (α, β) error rates.

## Files

- `python/wald_sprt.py` — Binomial `H₀: p = 0.4` vs
  `H₁: p = 0.6` at α = β = 0.05 (5 000 Monte Carlo trials).
  Realised Type-I ≈ 0.037, Type-II ≈ 0.036 (both ≤ target).
  Average sample number ≈ 37 obs under either hypothesis,
  vs a fixed-N test needing ~ 155.
- `r/wald_sprt.R` — `Sequential`, `gsDesign::sprt`,
  custom loop (R); `statsmodels.stats.sequential`,
  from-scratch (Python).

## When to use

- **Sequential inference on streaming data** — you can stop
  as soon as evidence is decisive.
- **Cost-sensitive experiments** — pharma dose-finding, QC
  sampling, adaptive trials.
- **Two simple hypotheses** or exponential family.
- **Rare-event testing** — SPRT decides quickly on decisive
  data.

## When NOT to use

- **Composite hypotheses** — extend via GSPRT / MSPRT
  (Siegmund), or use group-sequential / alpha-spending.
- **Correlated observations** — the LR requires
  independence; use adjusted increments.
- **Nuisance parameters** — profile / conditional SPRT.
- **You need a hard upper bound on n** — SPRT can run
  arbitrarily long; add a truncation rule.

## Assumptions & caveats

- **Simple-vs-simple** — for one-sided composite tests use
  worst-case LR.
- **Continued fraction over-shoot** — realised α, β are
  usually a bit BELOW nominal (Wald's inequalities are
  slightly conservative).
- **Truncation** — Anderson (1960) / Aroian truncated SPRTs
  bound the sample size.
- **Dependence structure** — mis-specified likelihood invalid-
  ates error control.

## Related in this repo

- `sequential-analysis`, `group-sequential-design`,
  `alpha-spending-lan-demets` — group-sequential extensions.
- `always-valid-inference` — modern e-value + confidence
  sequence framework.
- `sample-size-reestimation`, `conditional-power-futility`
  — adaptive trial cousins.
- `sequential-analysis` — general umbrella.

## Run

```
python techniques/wald-sprt/python/wald_sprt.py
Rscript techniques/wald-sprt/r/wald_sprt.R
```

**Refs:** Wald, A. "Sequential tests of statistical hypotheses." *Ann. Math. Statist.*, 16(2): 117-186, 1945; Wald, A. *Sequential Analysis*, Wiley, 1947.

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
