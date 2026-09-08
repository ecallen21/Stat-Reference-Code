# Semi-Markov Multi-State Model (Reference §47.47)

Foucher et al (2010). Extends Markov multi-state models by letting
the transition intensity depend on **sojourn time** (time since
entry to the current state), not calendar time:

    q_{ij}(t | u) = q_{ij}(u),   u = t − entry time.

Encodes duration-dependent hazards (e.g. accelerating relapse risk
the longer a patient stays in remission). Fitted here via
parametric Weibull sojourn + multinomial transition matrix.

## Files

- `python/semi_markov_multistate.py` — Weibull sojourn +
  Bernoulli next-state, right-censored MLE. Demo (n=800,
  shape=1.5, scale=3.0, P(1→2)=0.7, admin cens at t=5):
  MLE (shape, scale, P̂(1→2)) = (1.53, 3.10, 0.71).
- `r/semi_markov_multistate.R` — `SemiMarkov`, `msm`, `mstate`
  (R); custom (Python, no established package).

## When to use

- **Duration-dependent hazards** — sojourn ≠ constant.
- **Chronic disease progression** — e.g. remission → relapse rate
  climbs with time in remission.
- **Machine reliability** — wear-out phase (Weibull shape > 1).
- **Insurance claim durations** — actuarial multi-state models.

## When NOT to use

- **Time-since-origin dependence dominant** — use Cox-Markov
  multi-state.
- **Very small counts per transition** — parametric distributions
  overfit; fall back to nonparametric NPMLE.
- **Panel data with unknown transition times** — need
  interval-censored likelihood (Turnbull for each state).

## Assumptions & caveats

- **Weibull sojourn** is convenient; verify with residual QQ,
  Cox-Snell residuals, or LR test vs generalised-gamma.
- **Independent censoring** conditional on covariates.
- **Multinomial next state** conditional on leaving — often
  reasonable in AIDS/renal-transplant literature.
- **Identifiability** — need enough sojourns per transition to
  fit shape parameter; pool by strata otherwise.

## Related in this repo

- `markov-transition-models`, `multi-state-models` — Markov
  siblings.
- `illness-death-model`, `competing-risks`, `fine-gray`,
  `cure-models` — survival extensions.
- `kaplan-meier`, `nelson-aalen`, `parametric-survival` — one-state
  baselines.
- `piecewise-exponential-model`, `accelerated-failure-time`,
  `frailty-models` — related hazard specifications.

## Run

```
python techniques/semi-markov-multistate/python/semi_markov_multistate.py
Rscript techniques/semi-markov-multistate/r/semi_markov_multistate.R
```

**Refs:** Foucher, Y., Mathieu, E., Saint-Pierre, P., Durand, J.-F. & Daurès, J.-P. "A semi-Markov model based on generalized Weibull distribution with an illustration for HIV disease." *Biom J* 47(6): 825-833, 2005; Foucher, Y. et al. "A semi-Markov model with covariates for the study of the AIDS epidemic." *LIDA* 16: 316-338, 2010.

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
