# Minimum Description Length (Reference §34.8)

Rissanen (1978). Model selection by **shortest total code length**:

```
L_total(y, M)  =  L_model(M)  +  L_data(y | M).
```

## Two-part MDL

```
L_model  ≈  (k / 2) log n                       (parameter code)
L_data   =  − log p(y | θ̂_M)                    (data code)
```

Equivalent to **BIC** for regular parametric families.

## Normalised Maximum Likelihood (Rissanen 1996)

```
L_NML(y | M)  =  − log( p(y | θ̂(y)) / C(M) )
                 C(M) = Σ_y p(y | θ̂(y))          (complexity term).
```

Sharper, parameter-free, but computing `C(M)` is often hard.

## When to use

- **Model selection with a philosophical justification** — universal
  coding / stochastic complexity.
- **Comparison to BIC** — same asymptotics but different interpretation.
- **Model-order selection** in time-series / ARIMA.

## When NOT to use

- **Predictive performance is all that matters** — use CV / WAIC.
- **Complex non-parametric families** — coding-length calculation is
  intractable.

## Files

- `python/minimum_description_length.py` — two-part MDL on nested
  polynomial regressions. Demo: n=100, true cubic order.
  **Minimum L_total at order 3 (79.59)** matches true order; equal
  to BIC-selected model.
- `r/minimum_description_length.R` — `stats::BIC`, `minMDL` (R);
  `scipy` + custom NML (Python).

## Assumptions & caveats

- **Regular parametric families** — the `(k/2) log n` term is
  asymptotic.
- **Prior encoding** — different codes lead to slightly different
  penalties (BIC uses uniform prior on parameters).
- **NML** — computing `C(M)` requires summing over all data
  sequences; feasible only for small models.
- **Refined MDL** (Rissanen 2000) uses the Fisher information for a
  tighter penalty.

## Related in this repo

- `information-criteria` — BIC / AIC / DIC / WAIC family.
- `bayesian-model-comparison`, `bayes-factor` (adjacent) — Bayesian
  siblings.
- `cross-validation` — the predictive-accuracy alternative.
- `information-geometry` — the geometric side of MDL.

## Run

```
python techniques/minimum-description-length/python/minimum_description_length.py
Rscript techniques/minimum-description-length/r/minimum_description_length.R
```

**Refs:** Rissanen, J. "Modeling by shortest data description." *Automatica*, 1978; Rissanen, J. "Fisher information and stochastic complexity." *IEEE Trans IT*, 1996; Grünwald, P.D. *The Minimum Description Length Principle*, MIT Press, 2007.

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
