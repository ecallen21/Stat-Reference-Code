# Conditional Power / Futility Stopping (Reference §47.259)

Lan & Wittes (1988). Given interim data `Z_1` at information
fraction `t_1`, the CONDITIONAL POWER (CP) is the probability
of rejecting H₀ at the final look given the current data and
an ASSUMED true effect:

```
CP(δ) = 1 − Φ((z_α − Z_1 √t_1 − δ(1 − t_1)) / √(1 − t_1))
```

Common choices for `δ`: null (worst case), observed (Bayesian
flavour), or planned (best case). Rule of thumb: STOP FOR
FUTILITY if CP(observed) < 0.20.

## Files

- `python/conditional_power_futility.py` — Closed-form CP with
  a futility simulation showing Type-I preservation. Demo:
  planned drift 3.0, `t_1=0.5`. CP table across interim Z
  values (−1 to +2): at Z=1.0, CP(null)=0.24, CP(observed)=0.62,
  CP(planned)=0.86. Non-binding futility at CP<20% reduces
  Type-I from 0.024 to 0.020 (conservative).
- `r/conditional_power_futility.R` — `rpact::getConditionalPower`,
  `gsDesign::gsCP`, `gsDesign::gsBoundSummary` (R); rpact via
  reticulate, from-scratch (Python).

## When to use

- **Non-binding futility** — permit trial continuation past
  the threshold; no α inflation.
- **DSMB monitoring** — CP summarises the "chance of eventual
  success" more intuitively than a raw p-value.
- **Adaptive designs** — CP triggers sample-size
  re-estimation.

## When NOT to use

- **Binding futility** — needs α adjustment because the null
  region is being partitioned; use gsDesign / rpact with
  binding boundaries.
- **Efficacy stopping** — CP is a futility / decision tool;
  formal efficacy boundaries come from the group-sequential
  spending function.

## Assumptions & caveats

- **Assumed δ** matters enormously — always report CP under
  multiple δ (null / observed / planned).
- **Predictive power vs CP** — CP conditions on a single δ;
  predictive power averages over the posterior of δ (more
  informative when priors are credible).
- **Non-binding vs binding** — non-binding preserves α without
  any adjustment; binding requires re-derived boundaries.
- **Type-II inflation** — aggressive futility reduces power;
  balance the CP threshold to accepted power loss.

## Related in this repo

- `group-sequential-design` — the surrounding boundary
  machinery.
- `alpha-spending-lan-demets` — schedule-flexible boundaries.
- `sample-size-reestimation` — often triggered by low CP.

## Run

```
python techniques/conditional-power-futility/python/conditional_power_futility.py
Rscript techniques/conditional-power-futility/r/conditional_power_futility.R
```

**Refs:** Lan, K.K.G. and Wittes, J. "The B-value: A tool for monitoring data." *Biometrics*, 44(2): 579-585, 1988.

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
