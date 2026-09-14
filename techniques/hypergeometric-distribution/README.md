# Hypergeometric Distribution (Reference §47.391)

Sampling WITHOUT replacement. `X` = number of successes in
`n` draws from a finite population of `N` items with `K`
successes:

```
P(X = k) = C(K, k) · C(N − K, n − k) / C(N, n)
E[X] = n · K/N
Var  = n · (K/N) · ((N − K)/N) · ((N − n)/(N − 1))
```

The last factor `(N − n)/(N − 1)` is the FINITE-POPULATION
CORRECTION — the difference from a Binomial(n, K/N).

## Files

- `python/hypergeometric_distribution.py` — `N=50, K=20,
  n=10`. Empirical mean 4.01 (theory 4.00), variance 1.97
  (theory 1.96), PMF values match to 3 decimals across the
  support.
- `r/hypergeometric_distribution.R` — `stats::dhyper` /
  `rhyper`, `fisher.test` (R); `scipy.stats.hypergeom`,
  from-scratch (Python).

## Where else it appears in this repo

- `fisher-exact`, `fisher-exact-barnard` — 2×2 tables are
  hypergeometric under the null.
- `capture-recapture` — Petersen / Chapman-Petersen
  estimators use hypergeometric sampling model.
- `permutation-tests` — many permutation tests factor via
  the hypergeometric.
- `binomial-test`, `wilson-score-interval-proportion` —
  with-replacement (large-N) analogues.
- `gwas`, `gsea` — hypergeometric enrichment testing.

## Assumptions & caveats

- **Small n/N ratio** — Binomial is a good approximation
  when `n / N ≤ 0.1`.
- **Support** — `max(0, n + K − N) ≤ k ≤ min(K, n)`.
- **Symmetry** — swapping `K ↔ n` and `k ↔ k` gives the same
  probability (Vandermonde identity).
- **Multivariate extension** — categorical without
  replacement generalises to the multivariate
  hypergeometric.

## Run

```
python techniques/hypergeometric-distribution/python/hypergeometric_distribution.py
Rscript techniques/hypergeometric-distribution/r/hypergeometric_distribution.R
```

**Refs:** Feller, W. *An Introduction to Probability Theory and Its Applications*, Vol I, 3rd ed., Wiley, 1968.

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
