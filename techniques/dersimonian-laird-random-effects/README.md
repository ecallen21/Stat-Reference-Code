# DerSimonian-Laird Random-Effects (Reference §47.269)

DerSimonian & Laird (1986). Classical random-effects estimator
for pooling `K` studies with effect estimates `y_k` and
within-study variances `v_k`. Between-study variance τ² is
estimated via method-of-moments from Cochran's Q:

```
w_k^FE = 1 / v_k
Q = Σ w_k^FE (y_k − ȳ^FE)²
τ² = max(0, (Q − (K−1)) / (Σ w − Σ w² / Σ w))
w_k^RE = 1 / (v_k + τ²)
```

Wider CI than fixed-effect when heterogeneity is present.

## Files

- `python/dersimonian_laird_random_effects.py` — Formulas +
  simulated 10-study demo (true µ=0.5, τ=0.2). Q=65.5, I²=86%,
  τ̂²=0.023 (true 0.04). FE 95% CI (0.47, 0.54) narrower than
  RE (0.42, 0.62) — the RE interval acknowledges heterogeneity.
- `r/dersimonian_laird_random_effects.R` — `metafor::rma(method='DL')`,
  `meta::metagen`, `robumeta` (R); PythonMeta, from-scratch
  (Python).

## When to use

- **Heterogeneous studies** — different populations, protocols,
  or endpoints where a common true effect is implausible.
- **Default choice** in medical / social-science meta-analyses
  (Cochrane Handbook baseline).
- **Small-K reporting** — combine with HKSJ adjustment for
  better CI coverage.

## When NOT to use

- **Very small K (≤ 3)** — DL τ² estimate is unreliable;
  Bayesian meta-analysis with informative τ prior preferred.
- **Truly homogeneous studies** — fixed-effect is more
  efficient (test with Q, I², visual diagnostics).
- **Aggregate + IPD mixture** — a two-stage IPD analysis is
  more efficient.

## Assumptions & caveats

- **Normal RE assumption** — θ_k ~ N(µ, τ²); heavy-tailed
  θ_k needs t-based RE or Bayesian analogues.
- **DL τ² can under-estimate** — REML, Paule-Mandel, or
  Sidik-Jonkman estimators are less biased in small K.
- **Wald z-CI is anti-conservative** — HKSJ (t-quantile
  + adaptive variance) is the modern default for CI.
- **Publication bias** must be assessed separately (Egger,
  funnel plot, trim-and-fill).

## Related in this repo

- `hartung-knapp-sidik-jonkman` — CI adjustment for RE.
- `i-squared-heterogeneity` — heterogeneity diagnostics.
- `egger-test-publication-bias` — small-study effect test.
- `leave-one-out-meta`, `cumulative-meta-analysis` —
  sensitivity analyses.
- `network-meta-analysis` — multi-treatment extension.

## Run

```
python techniques/dersimonian-laird-random-effects/python/dersimonian_laird_random_effects.py
Rscript techniques/dersimonian-laird-random-effects/r/dersimonian_laird_random_effects.R
```

**Refs:** DerSimonian, R. and Laird, N. "Meta-analysis in clinical trials." *Controlled Clin. Trials*, 7(3): 177-188, 1986.

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
