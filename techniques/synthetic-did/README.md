# Synthetic Difference-in-Differences (Reference §35.10)

Arkhangelsky et al. (2021). Combines synthetic-control **unit
weights** (match treated unit's pre-period trajectory) with a DiD-style
**time weights** (equal weighting of pre vs post):

```
τ̂_SDID  =  argmin_τ  Σ_{i, t}  ω_i · λ_t · ( y_it − α_i − β_t − τ D_it )².
```

Improves over both plain DiD (which uses uniform unit weights) and
plain synthetic control (which uses uniform time weights).

## When to use

- **Single treated unit or a small treated set** with a rich control
  pool.
- **Heterogeneous unit trends** that violate the parallel-trends
  assumption of plain DiD.
- **Panel-level policy evaluations** — a state / country adopts a
  policy.

## When NOT to use

- **Many treated units + staggered adoption** — see `staggered-did`.
- **No similar control units** — the SC step can't build a good
  counterfactual.
- **Short pre-period** — unit weights become unstable.

## Files

- `python/synthetic_did.py` — compact simplex-projected SC unit
  weights + uniform time weights. Demo: 12 units, T = 20 (T_pre =
  12), heterogeneous linear trends per unit; true `τ = 2.0`. Result:
  **plain DiD = 1.12 (biased low), SDID = 2.46**; top SC weights
  concentrate on 2 units (0.49 + 0.31).
- `r/synthetic_did.R` — `synthdid` reference (R); `synthdid.py`
  (Python port).

## Assumptions & caveats

- **Simplex weights** — SC uses convex weights on controls; solved
  here by projected gradient.
- **Time weights** — Arkhangelsky's original solves an L2 problem;
  the compact demo uses uniform pre-weights.
- **Standard errors** — bootstrap over units.
- **Bias-variance trade-off** — SDID has lower variance than SC and
  lower bias than DiD in Monte Carlo (paper Fig 1-2).

## Related in this repo

- `diff-in-diff`, `staggered-did`, `event-study` — sibling designs
  (this batch).
- `synthetic-control` (adjacent) — the pure-SC alternative.
- `causal-forest` (if present) — HTE-oriented alternative.

## Run

```
python techniques/synthetic-did/python/synthetic_did.py
Rscript techniques/synthetic-did/r/synthetic_did.R
```

**Refs:** Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W. & Wager, S. "Synthetic difference in differences." *American Economic Review*, 2021; Abadie, A. "Using synthetic controls: feasibility, data requirements, and methodological aspects." *JEL*, 2021.

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
