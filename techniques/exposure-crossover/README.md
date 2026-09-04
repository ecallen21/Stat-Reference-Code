# Exposure Crossover / Drug-Drug Interaction (Reference §43.6)

Hennessy et al. (2014), Rothman (1976). **Within-person** design for
detecting drug-drug interactions: compare outcome rates during
periods of co-exposure (A + B) versus single exposure (A only,
B only) inside the same patient.

## Relative Excess Risk due to Interaction (RERI)

```
RERI = RR_AB − RR_A − RR_B + 1
```

- `RERI > 0` — super-additive interaction (A × B > sum of parts).
- `RERI = 0` — no additive interaction.
- `RERI < 0` — sub-additive interaction (antagonism).

## When to use

- **Drug-drug interaction** hypothesis generation from claims / EHR
  data.
- **Anaesthesia / oncology combinations** with hard-to-randomise
  pairings.
- **Post-marketing safety** when co-prescription is common.

## When NOT to use

- **Chronic co-medication** with insufficient discordant person-
  time.
- **Very rare events** — Poisson counts too small.
- **When linear-scale multiplicative interaction** is the target —
  use standard interaction terms in a Cox / logistic model.

## Files

- `python/exposure_crossover.py` — Poisson-rate RERI estimator
  from per-state person-time counts. Demo (n=5000, RR_A=1.5,
  RR_B=1.3, true RERI=1.0): recovered **RR_A 1.45, RR_B 1.33,
  RR_AB 2.72, RERI 0.94** — direction and magnitude match.
- `r/exposure_crossover.R` — `survival::clogit`, `gnm`,
  `epiR::epi.interaction` (R);
  `statsmodels.discrete.conditional_models.ConditionalLogit`,
  custom (Python).

## Assumptions & caveats

- **Constant hazards within person-periods** — Poisson assumption;
  extend to Cox for time-varying baselines.
- **Independent person-periods** within a patient — for recurrent
  events, use robust (cluster-sandwich) SEs.
- **Delta-method SE** for RERI — see Hosmer-Lemeshow 1992 or the
  R `epiR::epi.interaction`.
- **Discordant person-time** must exist for both single-drug and
  co-exposure states, or RERI is undefined.

## Related in this repo

- `sccs-self-controlled` — within-person analogue for a single
  exposure.
- `case-crossover` (if present) — matched-pair variant.
- `prescription-sequence-symmetry` — ordering-based interaction
  signal.

## Run

```
python techniques/exposure-crossover/python/exposure_crossover.py
Rscript techniques/exposure-crossover/r/exposure_crossover.R
```

**Refs:** Hennessy, S., Leonard, C.E., Palumbo, C.M., Shi, X., & Bilker, W.B. "Quality of Medicaid and Medicare data obtained through Centers for Medicare and Medicaid Services (CMS)." *Journal of American Health Economics*, 2014; Rothman, K.J. "The estimation of synergy or antagonism." *American Journal of Epidemiology*, 1976.

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
