# Hartung-Knapp-Sidik-Jonkman (Reference §47.270)

Hartung & Knapp (2001); Sidik & Jonkman (2002). Improved
inference for random-effects meta-analysis. Instead of
z-quantiles on the DL variance, use t-quantiles with `K−1`
df AND a variance estimator that adapts to observed
heterogeneity:

```
V_HK = (1/(K−1)) Σ w_k^RE (y_k − ȳ^RE)² / Σ w_k^RE
CI  = ȳ^RE ± t_{K−1, 1−α/2} · √V_HK
```

Recommended default for small K (< 20) and moderate
heterogeneity (Cochrane Handbook v6+).

## Files

- `python/hartung_knapp_sidik_jonkman.py` — Both DL-z and HKSJ
  CIs plus a Type-I simulation. Demo: K=5,10,30. Type-I under
  H₀ over 5000 sims of K=8 heterogeneous studies: DL-z=0.092
  (inflated by 84% relative to nominal), HKSJ=0.052 (very
  close to nominal 0.05).
- `r/hartung_knapp_sidik_jonkman.R` — `metafor::rma(test='knha')`,
  `meta::metagen(hakn=TRUE)`, `robumeta` (R); PythonMeta,
  from-scratch (Python).

## When to use

- **Random-effects meta-analyses with K < 20** — HKSJ is the
  Cochrane-recommended default.
- **Moderate heterogeneity** — the variance-adaptive component
  matters most here.
- **Regulatory-level rigor** — HKSJ has better Type-I
  properties than the classic DL-z.

## When NOT to use

- **Large K (≥ 50) and homogeneous studies** — DL-z and HKSJ
  converge; the simpler CI is fine.
- **Extreme τ² = 0 estimates** — HKSJ can widen CIs
  paradoxically; report both intervals.
- **Bayesian meta-analysis** — HKSJ is a frequentist fix;
  posterior credible intervals are the Bayesian analogue.

## Assumptions & caveats

- **Small-K conservatism** — HKSJ can be TOO conservative
  when τ² is truly zero; monitor coverage in simulation
  studies.
- **Rare implementation options** — `metafor::rma` needs
  `test='knha'`; `meta::metagen` needs `hakn=TRUE`.
- **REML + HKSJ** is the current Cochrane-preferred combo
  (REML for τ² + HKSJ for CI).
- **Adjusted t distribution** — the HKSJ variant with
  Röver-corrected df is even better for very small K.

## Related in this repo

- `dersimonian-laird-random-effects` — the base estimator
  HKSJ wraps.
- `i-squared-heterogeneity`, `leave-one-out-meta` —
  companion diagnostics.

## Run

```
python techniques/hartung-knapp-sidik-jonkman/python/hartung_knapp_sidik_jonkman.py
Rscript techniques/hartung-knapp-sidik-jonkman/r/hartung_knapp_sidik_jonkman.R
```

**Refs:** Hartung, J. and Knapp, G. "On tests of the overall treatment effect in meta-analysis with normally distributed responses." *Stat. Med.*, 20(12): 1771-1782, 2001; Sidik, K. and Jonkman, J.N. "A simple confidence interval for meta-analysis." *Stat. Med.*, 21(21): 3153-3159, 2002.

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
