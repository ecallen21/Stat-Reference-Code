# Illness-Death Multi-State Model (Reference §11.31)

The classical 3-state Markov model for disease progression:

    (0) Healthy ---q_01---> (1) Ill ---q_12---> (2) Dead
         \___________________q_02_____________________^

with transition intensities `q_01, q_02, q_12`. Dead is absorbing.

## Forward Kolmogorov

    P'(t) = P(t) · Q
    P(t) = exp(Q · t)

Fit by MLE from observed sojourn times + transition counts (exact
time) or from panel snapshots (msm library).

## Files

- `python/illness_death_model.py` — exact-time MLE + `expm(Q t)` for
  transition probabilities from scratch. Demo (n=400, T_max=25,
  true q = (0.10, 0.05, 0.15)): recovers q̂ = (0.093, 0.054,
  0.145); P(alive-healthy | start healthy, t=10) = 0.223, ill
  0.223, dead 0.554.
- `r/illness_death_model.R` — `msm`, `mstate`, `eha` (R);
  `lifelines` multi-state extensions, from-scratch (Python).

## When to use

- **Disease natural-history modelling** — cognitive decline,
  cancer progression, chronic disease.
- **Panel data with snapshots at exam visits** — msm supports.
- **Competing / recurrent events** — extend Q with more states.
- **Sojourn-time / survival with intermediate outcomes** — reveals
  whether "ill" precedes death or not.

## When NOT to use

- **Continuous outcomes** — use longitudinal / functional models.
- **Non-Markov transitions** — extend to semi-Markov or use
  frailty models.
- **State labels ambiguous** — need clear operational definition
  of "ill".

## Assumptions & caveats

- **Markov property** — intensities depend only on current state.
- **Time-homogeneous Q** by default; extend to piecewise-constant
  or covariate-dependent (Cox-Markov).
- **Panel misclassification** — msm supports hidden-Markov
  extensions.
- **Standard errors** — from observed information; likelihood
  ratio for testing Q constraints.

## Related in this repo

- `multi-state-models`, `markov-transition-models`, `cox-ph`,
  `frailty-models`, `competing-risks`, `hmm` — survival family.
- `parametric-survival`, `piecewise-exponential-model` —
  parametric siblings.
- `state-space-kalman` — different "state" concept.

## Run

```
python techniques/illness-death-model/python/illness_death_model.py
Rscript techniques/illness-death-model/r/illness_death_model.R
```

**Refs:** Kalbfleisch, J.D. & Prentice, R.L. *The Statistical Analysis of Failure Time Data*, 2nd ed., Wiley, 2002 (ch 8); Andersen, P.K. & Keiding, N. "Multi-state models for event history analysis." *Stat Methods Med Res*, 11(2): 91-115, 2002.

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
