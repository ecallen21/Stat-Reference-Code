# Confirmatory Factor Analysis (Reference §19.5)

**Confirmatory** — you specify the loading pattern in advance and test whether it fits. Contrast with **exploratory factor analysis** (`exploratory-factor-analysis`) which discovers the structure from the data.

## Model

```
x_i = Λ · η_i + ε_i
η_i ~ N(0, Φ)                 factor covariance
ε_i ~ N(0, Θ)                 error covariance (usually diagonal)
```

## Implied covariance

```
Σ(θ) = Λ Φ Λᵀ + Θ
```

## ML estimation

Minimize the ML discrepancy:

```
F_ML(θ) = tr(S Σ⁻¹) + log|Σ| − log|S| − p
χ²      = (n − 1) · F_ML         ~ χ²(df) if model is correct
df      = p(p + 1)/2 − (# free parameters)
```

## Fit indices

| Index | Rule of thumb |
|-------|---------------|
| CFI   | ≥ 0.95 good, ≥ 0.90 acceptable |
| RMSEA | ≤ 0.06 close fit, ≤ 0.08 acceptable |
| SRMR  | ≤ 0.08 acceptable, ≤ 0.05 good |
| χ² p-value | large (> 0.05) — but sensitive to `n` |

## Files

- `python/cfa_confirmatory_factor.py` — from-scratch ML CFA with L-BFGS on the covariance discrepancy + CFI, RMSEA, SRMR fit indices. Demo (n = 500, 2 factors × 3 indicators, factor correlation 0.4): recovers loadings within ~0.05 of truth; factor cor = 0.395; CFI 0.994, RMSEA 0.039, SRMR 0.024 — all "good".
- `r/cfa_confirmatory_factor.R` — `lavaan::cfa` (Rosseel; the canonical R implementation with rich model syntax, robust SEs, and modification indices).

## When to use

- **Test a pre-specified measurement model** — questionnaire subscales, competency clusters, cognitive-ability factors.
- **Measurement invariance** across groups / time — configural / metric / scalar equivalence.
- **First step of full SEM** — validate the measurement model before adding structural paths (see `path-analysis`).

## When to prefer EFA

- No strong prior on the loading pattern.
- Early stage of scale development.

## Assumptions & caveats

- **Multivariate normality** for the ML fit — use `lavaan::cfa(estimator = "MLR")` for robust ML under non-normality.
- **Identification**: fix at least one loading per factor to 1 (or fix factor variance to 1) for unique scale.
- **Sample size** — general rule 10–20 subjects per estimated parameter; more for weak / correlated factors.
- **Modification indices** — data-driven model modifications inflate Type-I error; report as sensitivity analysis, not confirmatory.

## Run

```
python techniques/cfa-confirmatory-factor/python/cfa_confirmatory_factor.py
Rscript techniques/cfa-confirmatory-factor/r/cfa_confirmatory_factor.R
```

**Refs:** Jöreskog, K.G. "Statistical analysis of sets of congeneric tests." *Psychometrika* 36(2), 109–133, 1971; Kline, R.B. *Principles and Practice of Structural Equation Modeling*, 4th ed., Guilford, 2015; Rosseel, Y. "lavaan: an R package for structural equation modeling." *J. Stat. Softw.* 48(2), 1–36, 2012.

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
