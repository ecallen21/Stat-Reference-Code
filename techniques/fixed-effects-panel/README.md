# Panel Data: Fixed Effects, Random Effects, Hausman (Reference §12.31, §12.32)

Panel data: repeated observations on the same **units** (subjects, firms, countries) over time. Three standard estimators.

## Between (BE)

OLS on unit means: `ȳ_i = X̄_i β + u_i`. Uses only cross-unit variation. Biased under any unit-level confounding.

## Within / Fixed Effects (FE)

Demean within each unit:

```
(y_it − ȳ_i) = (X_it − X̄_i)ᵀ β + (u_it − ū_i)
```

Absorbs **any** time-invariant unit-level confounder `b_i`. Cost: can't estimate coefficients of time-invariant regressors.

## Random Effects (RE)

Treat `b_i` as random with `Var(b_i) = τ²`. GLS estimator:

```
β̂_RE = (Xᵀ Ω⁻¹ X)⁻¹ Xᵀ Ω⁻¹ y      Ω block-diag σ²I + τ²J
```

Efficient **if** `b_i ⊥ X_it`. Under correlation, RE is inconsistent.

## Hausman test (1978)

```
H_0 : RE consistent (b_i ⊥ X)
H_a : RE inconsistent, FE consistent
Statistic:  (β̂_FE − β̂_RE)ᵀ (Cov_FE − Cov_RE)⁻¹ (β̂_FE − β̂_RE)  ~ χ²(k)
```

Reject `H_0` → use FE.

## Files

- `python/fixed_effects_panel.py` — between + within + random-effects + Hausman all from scratch. Demo (N = 100 units, T = 5, unit effect `b_i` correlated with `x`): FE recovers β = (1.20, −0.51) matching truth (1.2, −0.5); pooled OLS biased at 1.59; between severely biased at 2.40; Hausman χ² = 0.5, p = 0.78 (doesn't detect endogeneity here because RE's Swamy-Arora transform accidentally cleans it up on this small demo).
- `r/fixed_effects_panel.R` — `plm::plm(model = "within" / "random")` + `plm::phtest`.

## When to use each

- **FE** (default): unit-level confounding is possible and time-invariant.
- **RE**: unit-level heterogeneity but confidently uncorrelated with regressors (rare in observational data).
- **BE**: only when between-unit differences are the target.
- **Two-way FE** (unit + time): absorb both unit-invariant time shocks (macroeconomic year effects) and time-invariant unit effects.

## Related methods

- **First-differences** — equivalent to FE for T = 2; less efficient for T > 2 under classical assumptions.
- **Diff-in-diff** (`diff-in-diff`) — special case of two-way FE with a treatment-and-period interaction.
- **Correlated random effects (Mundlak-Chamberlain)** — halfway house between FE and RE.

## Assumptions & caveats

- **Strict exogeneity** — `E[u_it | X_i1, ..., X_iT] = 0`. Fails under lagged-dependent-variable dynamics; use dynamic panel methods (Arellano-Bond).
- **Balanced vs unbalanced** panels both work; unbalanced needs care with SE estimators.
- **Cluster-robust SEs** (see `sandwich-robust-se`) are standard for panel inference; classical SEs understate uncertainty when errors are serially correlated within unit.

## Run

```
python techniques/fixed-effects-panel/python/fixed_effects_panel.py
Rscript techniques/fixed-effects-panel/r/fixed_effects_panel.R
```

**Refs:** Mundlak, Y. "On the pooling of time series and cross section data." *Econometrica* 46(1), 69–85, 1978; Hausman, J.A. "Specification tests in econometrics." *Econometrica* 46(6), 1251–1271, 1978; Wooldridge, J.M. *Econometric Analysis of Cross Section and Panel Data*, 2nd ed., MIT Press, 2010.

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
