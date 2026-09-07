# Mediation -- Natural Direct / Indirect Effects (Reference §15.38)

VanderWeele (2015); Imai, Keele & Tingley (2010). Potential-outcomes
generalisation of Baron & Kenny mediation that handles
**exposure–mediator interaction** correctly.

## Decomposition

For exposure `T`, mediator `M`, outcome `Y`:

    TE  = E[Y(1, M(1))] − E[Y(0, M(0))]
    NDE = E[Y(1, M(0))] − E[Y(0, M(0))]   (natural direct effect)
    NIE = E[Y(1, M(1))] − E[Y(1, M(0))]   (natural indirect effect)
    TE  = NDE + NIE

Under **sequential ignorability** and **no exposure-induced M–Y
confounder**, both are point-identified.

## VanderWeele closed form (continuous M, continuous Y)

With `M = α₀ + α₁·T + α₂·C + ε_M` and
`Y = β₀ + β₁·T + β₂·M + β₃·T·M + β₄·C + ε_Y`:

    NDE = β₁ + β₃ · E[M(0) | C = c]
    NIE = (β₂ + β₃) · α₁

## Files

- `python/mediation_natural_effects.py` — closed-form NDE/NIE from
  scratch with T-M interaction. Demo (n=5000, true NDE=0.50, NIE=0.64,
  TE=1.14): recovered NDE=+0.505, NIE=+0.619, TE=+1.124, proportion
  mediated 55%.
- `r/mediation_natural_effects.R` — `mediation::mediate`,
  `medflex::neImpute/neWeight`, `regmedint::regmedint`, `paths` (R);
  DoWhy, causallib (Python).

## When to use

- **Mechanism decomposition** — how much of the treatment effect goes
  through a specific pathway.
- **Health-disparities decomposition** — direct effect of exposure vs
  indirect via mediators (e.g., education → income → health).
- **Trial design** — target-mediator strategies to boost NIE.

## When NOT to use

- **Exposure-induced M–Y confounding** — natural effects are no
  longer identified without additional assumptions; use
  **interventional (in)direct effects** or path-specific effects.
- **Multiple correlated mediators** — the single-mediator formulas
  above do not apply; use SEM or multivariate mediation
  frameworks.
- **Binary outcomes with rare events** — logistic-outcome-model
  NDE/NIE need the "rare disease" approximation or exact-marginal
  standardisation; parametric assumptions matter.

## Assumptions & caveats

- **Sequential ignorability** — no unmeasured T→Y or M→Y confounders
  after C.
- **Cross-world independence** — used to identify NDE (Imai et al.);
  untestable and controversial in some settings.
- **Correct mediator + outcome models** — misspecification of the
  T–M interaction bleeds into both NDE and NIE.
- **Bootstrap SEs** — analytic delta-method exists but bootstrap is
  simpler and standard in practice.

## Related in this repo

- `mediation-analysis` — classical Baron & Kenny approach.
- `path-analysis`, `cfa-confirmatory-factor` — SEM alternatives.
- `multilevel-mediation` — nested / cluster-mediation.
- `iptw`, `aipw-doubly-robust` — used to weight for identifying
  assumptions.

## Run

```
python techniques/mediation-natural-effects/python/mediation_natural_effects.py
Rscript techniques/mediation-natural-effects/r/mediation_natural_effects.R
```

**Refs:** VanderWeele, T.J. *Explanation in Causal Inference: Methods for Mediation and Interaction*, OUP, 2015; Imai, K., Keele, L. & Tingley, D. "A general approach to causal mediation analysis." *Psychological Methods*, 15(4): 309-334, 2010.

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
