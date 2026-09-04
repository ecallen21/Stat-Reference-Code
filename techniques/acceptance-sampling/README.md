# Acceptance Sampling (Reference §37.8 / §37.13)

Decide whether to accept or reject a lot based on a **sample** rather
than 100 % inspection.

## Single-sampling plan `(n, c)`

Inspect `n` items; accept the lot if defects `≤ c`, reject if `> c`.

## Key summaries

- **OC curve**: `P(accept | p) = Σ_{d=0..c} Bin(d; n, p)`.
- **Producer's risk α**: `P(reject | p = AQL)`.
- **Consumer's risk β**: `P(accept | p = LTPD)`.
- **AOQ** — average outgoing quality after 100 % rectification of
  rejected lots: `AOQ(p) = P_a(p) · p · (N − n)/N`.
- **AOQL** — max of AOQ(p) over p.
- **ATI** — average total inspection per lot.

## Double + sequential plans

- **Double**: two-stage decision with `(n_1, c_1)` and `(n_2, c_2, r_2)`.
- **Sequential**: Wald SPRT (see `sequential-analysis`).

## When to use

- **Destructive testing** where 100 % inspection is impossible.
- **Cost-limited inspection** in mass production.
- **Compliance** — Mil-Std-105E, ANSI Z1.4 acceptance-sampling
  standards.

## When NOT to use

- **Zero-defect required** — no sampling plan guarantees this.
- **Very small lots** — the hypergeometric distribution applies;
  binomial approximation biased.

## Files

- `python/acceptance_sampling.py` — OC curve + AOQ + AOQL + ATI for
  a single-sampling plan `(n=50, c=2, N=1000)`. Demo across
  `p ∈ {0.005, ..., 0.10}`:
  - **Producer's risk at AQL=0.01**: α = 0.014.
  - **Consumer's risk at LTPD=0.08**: β = 0.226.
  - **AOQL = 0.0257** at p = 0.040.
- `r/acceptance_sampling.R` — `AcceptanceSampling` (R reference);
  `scipy.stats.binom` + custom (Python).

## Assumptions & caveats

- **Binomial vs hypergeometric** — binomial OK if `n / N < 0.1`.
- **Independent defects** — clustered / correlated defects break
  binomial.
- **Compliance to standards** — use Mil-Std-105E / ANSI Z1.4 tables;
  the code above computes what those tables tabulate.
- **Non-conforming vs defective** — some standards distinguish.
- **Ongoing vs isolated lot** — switching rules (normal / tightened /
  reduced) apply for ongoing inspection.

## Related in this repo

- `sequential-analysis` — SPRT sampling plan.
- `six-sigma-methods`, `process-capability-indices` — quality-scoring
  siblings.
- `shewhart-control-charts` — process-monitoring sibling.
- `cusum-charts`, `ewma-charts` — small-shift monitoring.

## Run

```
python techniques/acceptance-sampling/python/acceptance_sampling.py
Rscript techniques/acceptance-sampling/r/acceptance_sampling.R
```

**Refs:** Dodge, H.F. & Romig, H.G. *Sampling Inspection Tables*, Wiley, 1959; Schilling, E.G. *Acceptance Sampling in Quality Control*, 2nd ed., CRC, 2009.

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
