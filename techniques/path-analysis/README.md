# Path Analysis (Reference §19.4)

Path analysis is **SEM restricted to observed variables** — a system of regressions linked by a directed acyclic graph (DAG) with no latent factors.

## Example

```
W → M → Y                (indirect path)
W ─────→ Y               (direct path)
```

Two connected regressions:

```
M = a · W + ε_M
Y = b · M + c · W + ε_Y
```

Total effect of `W` on `Y` = `c + a · b` (direct + indirect).

## Estimation

- **Recursive DAG** (no cycles): fit each equation by OLS — efficient. This is the case implemented below.
- **Non-recursive** (feedback loops): 2SLS / SEM ML (`lavaan::sem`).

## Files

- `python/path_analysis.py` — general OLS-per-equation fit + recursive `total_effect(source, target)` calculator that sums all path products. Demo (true `a = 0.6, b = 0.4, c = 0.3`): recovers `â = 0.60, b̂ = 0.37, ĉ = 0.34`; total 0.57 vs true 0.54.
- `r/path_analysis.R` — `lavaan::sem('M ~ W; Y ~ M + W', data = df)` for the standard R interface.

## When to use

- **Test a hypothesized DAG** of observed-variable causal relations.
- **Decompose total effects** into direct + all indirect paths.
- **Compare nested models** — remove a path, refit, compare fit indices (Δχ², CFI, BIC).

## Relation to nearby methods

- **Mediation analysis** — special case of path analysis with a single mediator.
- **SEM / CFA** — path analysis + latent variables.
- **Multilevel path analysis** — path analysis with clustered data (`lavaan::sem` with cluster arg).

## Fit indices

For nested / just-identified models the same χ², CFI, RMSEA, SRMR indices used in CFA (see `cfa-confirmatory-factor`) apply.

## Assumptions & caveats

- **DAG must be correct** — path coefficients are causal only under no unmeasured confounding along the paths.
- **Linear equations** in the demo; nonlinear/interactive paths need explicit interaction terms or a broader SEM.
- **Multivariate normality** for standard SEs; bootstrap for robust inference.
- **Identification**: recursive DAGs are always identified; non-recursive ones need instrument-like restrictions.

## Run

```
python techniques/path-analysis/python/path_analysis.py
Rscript techniques/path-analysis/r/path_analysis.R
```

**Refs:** Wright, S. "Correlation and causation." *J. Agric. Res.* 20(7), 557–585, 1921; Duncan, O.D. *Introduction to Structural Equation Models*, Academic, 1975; Kline, R.B. *Principles and Practice of Structural Equation Modeling*, 4th ed., Guilford, 2015.

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
