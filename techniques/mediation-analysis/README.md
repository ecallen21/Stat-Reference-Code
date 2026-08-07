# Mediation Analysis (Reference §15.15)

Decompose the total effect of `T → Y` into a **direct** effect and an **indirect** effect via a mediator `M`:

```
       M
      ↗   ↘
     T  →  Y            direct + indirect paths
```

## Baron-Kenny product method (1986)

Two regressions:

```
M = α + a · T + ε_M
Y = β_0 + c' · T + b · M + ε_Y
```

- **Direct effect** = `c'`
- **Indirect effect** = `a · b`
- **Total effect** = `a · b + c'`

## Modern causal formulation (Robins-Greenland 1992; Pearl 2001; Imai-Keele-Tingley 2010)

Using potential outcomes `Y(t, m)`:

```
Natural direct effect (NDE)   = E[Y(1, M(0)) − Y(0, M(0))]
Natural indirect effect (NIE) = E[Y(1, M(1)) − Y(1, M(0))]
Total                          = NDE + NIE
```

Under linear models with **no T×M interaction**, NDE = `c'` and NIE = `a · b` (matches Baron-Kenny). With interactions, use explicit standardization (`mediation::mediate`, `medflex`).

## Bootstrap for SEs / CIs

The product `a · b` isn't normally distributed even when `a` and `b` are; bootstrap gives correct percentile CIs.

## Files

- `python/mediation_analysis.py` — Baron-Kenny fit + nonparametric bootstrap CI on the indirect effect. Demo (true a = 0.8, b = 0.5, c' = 0.3): estimated a·b = 0.406 (true 0.4); bootstrap 95% CI (0.25, 0.57) covers truth.
- `r/mediation_analysis.R` — `mediation::mediate(model_m, model_y, treat = "T", mediator = "M")` for the full Imai-Keele-Tingley implementation with sensitivity analysis for the sequential-ignorability assumption.

## When to use

- **Mechanism decomposition** — "does treatment work through channel M, or does it act directly?"
- **Program evaluation** — decompose an intervention's effect into intermediate outcomes.
- **Clinical / psych research** — psychological therapies work by changing intermediate constructs.

## Assumptions

- **Sequential ignorability** — no unmeasured confounding of T→M **or** M→Y. This is the crucial (often violated) assumption. Sensitivity analysis with `mediation` package reports how strong an unobserved confounder would have to be to nullify the indirect effect.
- **No T×M interaction** for Baron-Kenny to equal NDE / NIE. For nonlinear or interactive models, standardize under the potential-outcomes framework.
- **Correct functional form** for both submodels.

## Alternatives

- **Multiple mediators** — `mediation::mediate` supports the joint case; VanderWeele 2015.
- **Multilevel mediation** (see `multilevel-mediation`) — clustered data.
- **Longitudinal mediation** — g-computation over time (see `tmle-doubly-robust`).
- **Instrumental-variable mediation** — when T→M is confounded.

## Run

```
python techniques/mediation-analysis/python/mediation_analysis.py
Rscript techniques/mediation-analysis/r/mediation_analysis.R
```

**Refs:** Baron, R.M. & Kenny, D.A. "The moderator-mediator variable distinction in social psychological research." *J. Pers. Soc. Psychol.* 51(6), 1173–1182, 1986; Robins, J.M. & Greenland, S. "Identifiability and exchangeability for direct and indirect effects." *Epidemiology* 3(2), 143–155, 1992; Pearl, J. "Direct and indirect effects." *UAI*, 2001; Imai, K., Keele, L. & Tingley, D. "A general approach to causal mediation analysis." *Psychol. Methods* 15(4), 309–334, 2010; VanderWeele, T. *Explanation in Causal Inference: Methods for Mediation and Interaction*, OUP, 2015.

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
