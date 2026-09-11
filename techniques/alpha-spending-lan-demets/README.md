# Alpha-Spending Function (Reference §47.258)

Lan & DeMets (1983). FLEXIBLE group-sequential boundaries:
pre-specify how much of the total α is spent by information
fraction `t = n_k / N`:

- **Pocock-like**: `α(t) = α · ln(1 + (e−1) t)`
- **O'BF-like**: `α(t) = 2 (1 − Φ(z_{α/2} / √t))`
- **Linear**: `α(t) = α · t`
- **Power**: `α(t) = α · t^ρ`

Advantages over classical Pocock / O'BF: interim looks do not
have to be evenly spaced, and `K` can change mid-trial (as
long as information fraction is respected).

## Files

- `python/alpha_spending_lan_demets.py` — closed-form spending
  functions plus MC calibration of per-look boundaries. Demo:
  `α=0.05`, `K=4`. OBF cumulative alpha at t=0.25/0.5/0.75/1.0
  is 0.0001/0.006/0.024/0.05; Pocock is 0.018/0.031/0.041/0.05
  (front-loaded); linear is 0.0125/0.025/0.0375/0.05. Boundaries
  at UNEVEN looks (0.15, 0.40, 0.80, 1.00) auto-adjust to
  preserve α.
- `r/alpha_spending_lan_demets.R` — `ldbounds::ldBounds`,
  `gsDesign::sfLDOF`/`sfLDPocock`, `rpact` (R); rpact via
  reticulate, from-scratch (Python).

## When to use

- **Interim schedule may drift** — the DSMB reschedules a
  look, N shifts; alpha-spending preserves α anyway.
- **Trials where information ≠ patients** — event-driven
  survival trials where the schedule is unpredictable.
- **Communication to regulators** — spending functions have
  a monotone budget interpretation that's easier to defend
  than fixed boundaries.

## When NOT to use

- **Very small `K` (≤ 2)** — Pocock / OBF classical boundaries
  are simpler and equivalent.
- **Adaptive rebuilds of the trial** — combine with CHW / RCI
  and preplanned re-weighting instead.

## Assumptions & caveats

- **Monotone α(t)** — spending must be non-decreasing; violated
  by ad-hoc adjustments.
- **Information fraction defined a priori** — for survival
  trials use expected events, not calendar time.
- **Boundary at t = 1** — OBF-like has a final boundary
  slightly above 1.96 (~2.04), a small power cost.

## Related in this repo

- `group-sequential-design` — the classic Pocock / OBF pair.
- `conditional-power-futility` — non-binding futility layer.
- `sample-size-reestimation` — combines with spending via
  CHW re-weighting.

## Run

```
python techniques/alpha-spending-lan-demets/python/alpha_spending_lan_demets.py
Rscript techniques/alpha-spending-lan-demets/r/alpha_spending_lan_demets.R
```

**Refs:** Lan, K.K.G. and DeMets, D.L. "Discrete sequential boundaries for clinical trials." *Biometrika*, 70(3): 659-663, 1983.

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
