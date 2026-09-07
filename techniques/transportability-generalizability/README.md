# Transportability / Generalizability (Reference §15.39)

Cole & Stuart (2010); Westreich et al. (2017). Standardises an
experimental causal effect from a **study** sample to a **target**
population whose distribution of effect modifiers differs.

## Inverse-odds weighting (IOW)

Let `S = 1` for study units, `S = 0` for target. For each study unit:

    w_i = P(S = 0 | X_i) / P(S = 1 | X_i)

Then the Hajek transported ATE is:

    Ê[Y(t) | S = 0] = Σ w_i · 1[T_i = t] · Y_i  /  Σ w_i · 1[T_i = t]
    ATE_target = Ê[Y(1) | S = 0] − Ê[Y(0) | S = 0]

## Files

- `python/transportability_generalizability.py` — logistic
  `P(S | X)` + IOW Hajek ATE. Demo (study n=2000 X~N(0,1), target
  n=4000 X~N(1.5,1); τ(x)=1+x; true study ATE 1.0, true target ATE
  2.5): recovers naive study +0.97 and transported target +2.59.
- `r/transportability_generalizability.R` — `generalize::transport`,
  `txshift` (R); `causallib.estimation.transport`, DoWhy (Python).

## When to use

- **RCT-to-real-world extrapolation** — trial enrolls
  younger/healthier patients than clinical use.
- **Multi-site generalisation** — one clinical site's ATE applied to
  a health system with a different case mix.
- **Policy evaluation** — evaluating a programme where distribution
  of moderators differs between pilot and rollout.

## When NOT to use

- **Non-overlap of X** — if the target has covariate combinations
  never seen in the study, IOW weights explode; use partial
  identification / bounds instead.
- **Effect modifiers not measured** — missing key modifiers biases
  the transported ATE; sensitivity analysis (VanderWeele bias
  factors) required.
- **Only in-study inference wanted** — no adjustment needed; report
  the study ATE.

## Assumptions & caveats

- **Randomisation of T** in the study (or unconfoundedness).
- **Positivity of S** — every covariate profile in the target has
  positive `P(S=1|X)`.
- **Exchangeability across S** given X — the outcome model is the
  same up to X-shift.
- **Include ALL effect modifiers in X** — missing ones give biased
  transport. Sensitivity via e-values on the transport step.

## Related in this repo

- `iecv-multisite` — internal-external cross-validation of prediction
  models across sites.
- `iptw`, `overlap-weighting`, `aipw-doubly-robust` — the within-
  study estimators.
- `covariate-shift-adaptation` — the ML sibling for prediction
  transport.
- `hte-uplift`, `causal-forest` — heterogeneous-effect approaches
  used before / with transportability.

## Run

```
python techniques/transportability-generalizability/python/transportability_generalizability.py
Rscript techniques/transportability-generalizability/r/transportability_generalizability.R
```

**Refs:** Cole, S.R. & Stuart, E.A. "Generalizing evidence from randomized clinical trials to target populations." *American Journal of Epidemiology*, 172(1): 107-115, 2010; Westreich, D. et al. "Transportability of trial results using inverse-odds of sampling weights." *American Journal of Epidemiology*, 186(8): 1010-1014, 2017.

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
