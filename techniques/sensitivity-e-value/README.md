# Sensitivity Analysis: E-value + Rosenbaum Bounds (Reference §15.14)

Assess robustness of an observational causal estimate to **unmeasured confounding**.

## VanderWeele-Ding E-value (2017)

Minimum risk ratio that an unmeasured confounder would need with **both** the treatment **and** the outcome — above and beyond measured confounders — to fully explain away the observed association.

```
E-value(RR) = RR + √(RR (RR − 1))                (for RR ≥ 1; invert first if RR < 1)
```

Report two E-values:

- **E-value(point estimate)** — how strong to explain away the estimate.
- **E-value(CI bound closer to null)** — how strong to render the CI compatible with null.

## Interpretation

- **E-value 1.5** — a modest unobserved confounder could nullify the finding; result is fragile.
- **E-value 3** — need a fairly strong confounder to explain away.
- **E-value 5+** — robust against most plausible unobserved confounders.

Always report **in context** — plausibility of an unmeasured confounder of that strength depends on domain.

## Extensions

- **Continuous outcomes**: approximate `RR ≈ exp(0.91 · d)` for standardized-mean-difference `d`.
- **Odds ratios (rare outcome)**: `RR ≈ OR`. **Common outcome**: `RR ≈ √OR`.
- **Hazard ratios**: for rare events, `RR ≈ HR`.

## Rosenbaum bounds

Alternative sensitivity framework for **matched-pair** designs:

```
Γ = ratio of odds of treatment for two units matched on observed covariates
```

Report the smallest Γ at which the treatment-effect p-value crosses 0.05. Γ = 2 means the odds of treatment could **double** for one member of any pair.

## Files

- `python/sensitivity_e_value.py` — E-value calculators for RR, OR (rare + common outcome), and continuous-effect approximation. Demos: RR = 2.0 (CI 1.4) → E-value point 3.41, CI bound 2.15; d = 0.5 → E-value 2.53; OR = 2.5 common outcome → E-value 2.54.
- `r/sensitivity_e_value.R` — `EValue::evalue(EValue::RR(...), lo, hi)`; `sensitivitymv::senmv` or `rbounds::psens` for Rosenbaum bounds.

## When to use

- **Any observational causal claim** — journal editors increasingly request an E-value alongside adjusted estimates.
- **Robustness reporting** for propensity-score, IPW, matching, or regression-adjustment analyses.
- **Comparing evidence strength** across observational vs randomized studies.

## Assumptions & caveats

- **E-value** is a summary of the sensitivity of the point estimate / CI, not a test of confounding.
- **Multiplicative confounder** implicitly assumed; VanderWeele-Ding also discusses joint conditioning.
- **Rosenbaum bounds** for pairs, not for weighted / stratified analyses (use `sensitivitymv` extensions there).
- **Report both** the point and the CI-bound E-value.

## Run

```
python techniques/sensitivity-e-value/python/sensitivity_e_value.py
Rscript techniques/sensitivity-e-value/r/sensitivity_e_value.R
```

**Refs:** Rosenbaum, P.R. *Observational Studies*, 2nd ed., Springer, 2002; VanderWeele, T.J. & Ding, P. "Sensitivity analysis in observational research: introducing the E-value." *Ann. Intern. Med.* 167(4), 268–274, 2017; Ding, P. & VanderWeele, T.J. "Sensitivity analysis without assumptions." *Epidemiology* 27(3), 368–377, 2016.

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
