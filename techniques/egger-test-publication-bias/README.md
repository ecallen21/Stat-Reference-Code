# Egger Test for Publication Bias (Reference §47.271)

Egger, Davey Smith, Schneider & Minder (1997). Regression-based
funnel-plot asymmetry test:

```
y_k / SE_k = a + b · (1 / SE_k)         (WLS)
```

A non-zero intercept `a` indicates SMALL-STUDY EFFECTS
(publication bias, methodological quality gradient, or true
heterogeneity by size).

## Files

- `python/egger_test_publication_bias.py` — WLS regression on
  standardised effect vs precision. Demo: (A) no bias with 20
  studies gives intercept ≈ 0 and p > 0.05 (do not reject);
  (B) with inflated small studies, intercept > 0 and p < 0.05
  (reject H₀). Funnel-plot snapshot shows small-SE studies
  clustered vs large-SE studies scattered high.
- `r/egger_test_publication_bias.R` — `metafor::regtest`,
  `meta::metabias(method='linreg')`, `metasens` (R);
  PythonMeta, from-scratch (Python).

## When to use

- **Continuous / smd effects** — Egger's original setting.
- **After primary meta-analysis** to check funnel-plot
  symmetry.
- **≥ 10 studies** — with fewer studies the test is
  underpowered.

## When NOT to use

- **OR / RR outcomes** — Egger has anti-conservative Type-I
  for rare events; use Peters test or Harbord test instead.
- **Very few studies** (< 10) — power is too low to detect
  bias.
- **Confounded by heterogeneity** — a non-zero intercept can
  come from real between-study heterogeneity in effect size,
  not just publication bias.

## Assumptions & caveats

- **Linear bias model** — Egger assumes bias is linear in
  precision; alternatives (Peters, Macaskill, PET-PEESE)
  make different assumptions.
- **Interpretation** — a significant Egger test does NOT
  prove publication bias; it flags funnel-plot asymmetry which
  can have many causes.
- **Trim-and-fill** provides a companion visual + adjusted
  estimate.
- **Report the ACTUAL funnel plot** alongside the test — the
  plot often tells more than the p-value.

## Related in this repo

- `dersimonian-laird-random-effects`,
  `hartung-knapp-sidik-jonkman` — the underlying meta-analysis.
- `trim-fill` — related bias-adjustment method.
- `leave-one-out-meta`, `cumulative-meta-analysis` — other
  sensitivity analyses.
- `mr-egger-pleiotropy` — Egger regression for Mendelian
  randomisation.

## Run

```
python techniques/egger-test-publication-bias/python/egger_test_publication_bias.py
Rscript techniques/egger-test-publication-bias/r/egger_test_publication_bias.R
```

**Refs:** Egger, M., Davey Smith, G., Schneider, M. and Minder, C. "Bias in meta-analysis detected by a simple, graphical test." *BMJ*, 315(7109): 629-634, 1997.

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
