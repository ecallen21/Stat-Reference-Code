# Pearl Front-Door Criterion (Reference §15.x extra)

Identifies the **causal effect of `T` on `Y`** through a mediator `M`, even
when `T` and `Y` are confounded by an **unmeasured** `U`, provided:

- `T → M` (T affects M),
- `M → Y` (M affects Y — no direct T → Y path),
- `U` does not affect `M` directly,
- no unblocked back-door from `M` to `Y`.

## The formula

```
P(Y | do(T = t)) = Σ_m P(M = m | T = t) · Σ_{t'} P(Y | T = t', M = m) · P(T = t')
```

Interpretation:

1. **First stage**: `P(M | T = t)` — how the mediator responds to treatment (identified because `T → M` has no back-door).
2. **Second stage**: `Σ_{t'} P(Y | T = t', M = m) · P(T = t')` — the outcome given `M`, back-door-adjusted for `T` (which blocks the U path since `T ⊥ Y | M, U` conditioned appropriately).
3. **Combine**: chain them via the mediator.

## Contrast with the back-door criterion

- **Back-door**: adjust for a sufficient set of measured confounders. Requires *knowing* and *measuring* them.
- **Front-door**: exploits a mediator to bypass unmeasured confounding. Rarely applicable in practice — needs a clean mediator path with no leaks.

## Classic example (Pearl 2000)

Smoking → tar deposits → lung cancer. If smoking / cancer are confounded by an unmeasured genetic factor, and tar is a full mediator with no direct genetic effect on tar and no non-tar path from smoking to cancer, the front-door identifies the causal effect from observational data.

## When to use

- **Full-mediation** situations with an unmeasured confounder between exposure and outcome and where a clean mediator was measured.
- **Sensitivity analysis** — compare the front-door estimate to back-door estimates that assume no unmeasured confounding.
- **Pedagogy** — the front-door is the textbook example of identification impossible with covariate adjustment alone.

## Files

- `python/front_door_criterion.py` — closed-form binary-`(T, M, Y)` front-door estimator. Demo (n = 20 000, unmeasured `U → T` and `U → Y`, `M` mediates `T → Y`, true ATE 0.34): front-door ATE = +0.337 vs true 0.342 (recovery); naive `E[Y|T=1] − E[Y|T=0]` = +0.504 (biased upward by U).
- `r/front_door_criterion.R` — `dagitty::adjustmentSets`, `causaleffect::causal.effect`, `bnlearn::query`; Python `DoWhy` for identify-and-estimate.

## Assumptions & caveats

- **Full mediation is strong** — any direct `T → Y` path invalidates the estimator; sensitivity analysis rarely reassures.
- **U → M leaks** invalidate the first stage; check the DAG carefully.
- **Continuous variables** need semi-parametric estimation — sequential regression / plug-in for `E[M|T]` and `E[Y|T, M]` (see `DoWhy`'s implementation).
- **Sample size** — the two-step estimator has larger variance than back-door adjustment; use the bootstrap for SEs.
- **Model dependence** — a mis-specified conditional distribution (e.g. `P(Y | T, M)`) inherits its bias into the estimand.

## Related in this repo

- `mediation-analysis` — Baron-Kenny / natural direct+indirect effects (assume no unmeasured T–Y confounder).
- `iv-2sls`, `mendelian-randomization` — instrument-based identification (different DAG).
- `propensity-score-matching`, `inverse-probability-weighting`, `tmle-doubly-robust` — back-door adjustment methods when all confounders are measured.
- `sensitivity-e-value` — quantify robustness to unmeasured confounding when neither back-door nor front-door applies cleanly.

## Run

```
python techniques/front-door-criterion/python/front_door_criterion.py
Rscript techniques/front-door-criterion/r/front_door_criterion.R
```

**Refs:** Pearl, J. *Causality: Models, Reasoning, and Inference*, 2nd ed., Cambridge UP, 2009; Pearl, J. "Causal diagrams for empirical research." *Biometrika* 82(4), 669–688, 1995; Hernán, M.A. & Robins, J.M. *Causal Inference: What If*, Chapman & Hall/CRC, 2020.

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
