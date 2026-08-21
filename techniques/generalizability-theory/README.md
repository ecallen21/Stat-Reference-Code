# Generalizability Theory (Reference §22.10)

Extends classical reliability by decomposing observed-score variance into **multiple facets** (raters, items, occasions, forms, ...). Cronbach-Gleser-Nanda-Rajaratnam (1972).

## Two-stage workflow

- **G-study** — estimate variance components in a designed dataset (persons × items × raters × ...). Uses random-effects ANOVA.
- **D-study** — forecast reliability for a proposed measurement design (different number of items / raters / occasions) using the G-study components.

## Variance components (p × i design)

```
sigma²_p      person (target of measurement)
sigma²_i      item (source of variance)
sigma²_pi     residual + person × item interaction
```

## G and Phi coefficients

- **G coefficient** — reliability for **relative** decisions (rankings):

```
ρ² = σ²_p / (σ²_p + σ²_pi / n_i)
```

- **Phi coefficient** — reliability for **absolute** decisions (cutoff-based):

```
Φ = σ²_p / (σ²_p + (σ²_i + σ²_pi) / n_i)
```

Absolute decisions include item main-effect variance because raw score levels matter.

## Files

- `python/generalizability_theory.py` — one-facet crossed p × i design + G, Φ + D-study forecast for different `n_items`. Demo (n_p = 100, n_i = 8): σ²_p = 2.02, σ²_i = 0.22, σ²_pi = 0.47; G = 0.97, Φ = 0.96. D-study shows G rises from 0.95 (4 items) to 0.99 (32 items).
- `r/generalizability_theory.R` — `gtheory::gstudy` / `dstudy`.

## When to use

- **Rater-based scoring** — dissertation grading, medical-image reads, essay scoring.
- **Multi-visit measurement** — patient-reported outcomes across occasions.
- **Test-form comparability** across forms and administrations.
- **Comprehensive reliability design** — better than α when there are multiple systematic sources.

## Extensions

- **Multiple facets** (p × i × r, p × (i:r), etc.) — more variance components.
- **Nested facets** — raters nested in items, etc.
- **Random vs fixed facets** — treatment of each facet affects the D-study denominators.

## Assumptions & caveats

- **Balanced design** for ANOVA-style estimation; unbalanced use REML.
- **Random-effects sampling** — items / raters sampled from a population; treat as fixed if selected purposefully.
- **Interpretation**: G coefficient ~ α for the p × i case; Phi is stricter (includes item main effect).

## Run

```
python techniques/generalizability-theory/python/generalizability_theory.py
Rscript techniques/generalizability-theory/r/generalizability_theory.R
```

**Refs:** Cronbach, L.J., Gleser, G.C., Nanda, H. & Rajaratnam, N. *The Dependability of Behavioral Measurements*, Wiley, 1972; Brennan, R.L. *Generalizability Theory*, Springer, 2001.

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
