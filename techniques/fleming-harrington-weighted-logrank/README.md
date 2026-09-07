# Fleming-Harrington Weighted Log-Rank (Reference §11.29)

Fleming & Harrington (1991). Generalises the log-rank test to
**weight** event times, gaining power when the treatment effect is
concentrated in time.

## G(ρ, γ) family

    w(t) = Ŝ(t⁻)^ρ · (1 − Ŝ(t⁻))^γ

    G(0, 0)  = classical log-rank
    G(1, 0)  = Peto-Peto / Prentice (early emphasis)
    G(0, 1)  = late emphasis
    G(1, 1)  = middle emphasis

Statistic:

    Z = Σ w(t) (O₁(t) − E₁(t)) / √(Σ w²(t) V(t))   ~   N(0, 1)

## Files

- `python/fleming_harrington_weighted_logrank.py` — G(ρ, γ) from
  scratch using pooled KM for weights. Demo (n=200 per arm,
  delayed treatment benefit at t=15): G(0,0) Z=1.79 p=0.073;
  G(0,1) Z=2.37 p=0.018 (correctly more powerful for late
  divergence); G(1,0) Z=1.06 p=0.29 (Peto over-weights early,
  loses power here).
- `r/fleming_harrington_weighted_logrank.R` — `survival::survdiff
  (rho=, gamma=)`, `FHtest`, `nph`, `survRM2` (R); from-scratch,
  `lifelines.statistics.multivariate_logrank_test` (Python).

## When to use

- **Delayed or crossing hazards** — immuno-oncology arms with
  delayed onset.
- **Sensitivity-analysis complement** — G(0,0) as primary + G(0,1)
  / max-combo as sensitivity (FDA-friendly).
- **Pre-specified alternative shape** — early / late / middle
  benefit expected on biological grounds.

## When NOT to use

- **Proportional hazards** — G(0,0) is optimal; anything else
  costs power.
- **Very small n** — the weighted variance formula requires
  enough events; consider permutation.
- **After the fact / cherry-picking** — pre-specify ρ, γ or use
  a max-combo test (nph package) with appropriate error control.

## Assumptions & caveats

- **Non-informative censoring** — same as any log-rank.
- **Weight based on POOLED KM** — protects vs post-hoc weighting.
- **Max-combo tests** (Karrison 2016) combine G(0,0), G(0,1),
  G(1,0), G(1,1) with correct family-wise error.
- **Interpretation** — a significant G(0,1) says treatment effect
  is late; does not imply HR is constant elsewhere.

## Related in this repo

- `log-rank-test`, `cox-ph`, `cox-time-varying`, `cure-models`,
  `rmst` — survival cousins.
- `sensitivity-e-value`, `sequential-analysis` — sensitivity /
  monitoring siblings.

## Run

```
python techniques/fleming-harrington-weighted-logrank/python/fleming_harrington_weighted_logrank.py
Rscript techniques/fleming-harrington-weighted-logrank/r/fleming_harrington_weighted_logrank.R
```

**Refs:** Fleming, T.R. & Harrington, D.P. *Counting Processes and Survival Analysis*, Wiley, 1991; Harrington, D.P. & Fleming, T.R. "A class of rank test procedures for censored survival data." *Biometrika*, 69(3): 553-566, 1982.

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
