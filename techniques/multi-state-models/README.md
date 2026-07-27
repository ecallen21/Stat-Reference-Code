# Multi-State Models: Illness-Death, Transition Hazards, State Occupation (Reference §11.27; also covers §11.52)

For processes with **multiple states** connected by permitted transitions, multi-state models generalize survival analysis. The canonical example is the **illness-death** model:

```
   healthy (0)  ──α_01(t)──▶  ill (1)  ──α_12(t)──▶  dead (2)
       │                                              ▲
       └─────────── α_02(t) ─────────────────────────┘
```

Each arrow has its own **transition intensity** `α_hk(t)`. The state-occupation vector `p(t) = (P_0(t), P_1(t), P_2(t))` gives probabilities of being in each state at time t, computed by product-integrating the intensity matrix.

## What's implemented here

- **Per-transition Nelson-Aalen cumulative hazards** `A_hk(t)` — one per arrow.
- **Illness-death P₀(t)** via `exp(−(A_01(t) + A_02(t)))`.
- Notes on how to extend to `P_1(t)`, `P_2(t)` (requires the full Aalen-Johansen product-integral; use R's `mstate::probtrans` for production).

## When to use

- Disease progression models (healthy → early stage → advanced → death).
- Multi-arm treatment paths (on-treatment → discontinued → death).
- Reversible processes (e.g. remission ↔ relapse) — extend transition matrix to include backward arrows.

Not to be confused with:

- **Competing risks** ([`competing-risks`](../competing-risks)) — a degenerate multi-state where all subjects start in the same state and transitions go to K absorbing states.
- **Recurrent events** ([`recurrent-events`](../recurrent-events)) — same event type happening repeatedly (a single-state process with self-loops).

## Files

- `python/multi_state_models.py` — Nelson-Aalen transition hazards per arrow + illness-death `P_0(t)` calculation.
- `r/multi_state_models.R` — pointer to `mstate` and `msm` packages, which are the authoritative multi-state implementations.

## Assumptions

- **Markov**: transition intensity depends only on the current state and time (or, in a semi-Markov model, time since entering the current state).
- Independent right-censoring within each transition.
- **Non-Markov extensions** (transition depends on entire history) require more elaborate machinery — not implemented.

## Run

```
python techniques/multi-state-models/python/multi_state_models.py
Rscript techniques/multi-state-models/r/multi_state_models.R
```

**Refs:** Andersen, P.K., Borgan, Ø., Gill, R.D. & Keiding, N. *Statistical Models Based on Counting Processes*, Springer, 1993 (Ch. IV); Putter, H., Fiocco, M. & Geskus, R.B. "Tutorial in biostatistics: competing risks and multi-state models." *Stat. Med.* 26(11), 2389–2430, 2007; de Wreede, L.C., Fiocco, M. & Putter, H. "mstate: An R package for the analysis of competing risks and multi-state models." *J. Stat. Soft.* 38(7), 1–30, 2011.

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
