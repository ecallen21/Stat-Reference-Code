# Pareto Distribution (Reference §47.393)

Pareto (1897). Type-I:

```
f(x; α, x_m) = α · x_m^α / x^(α+1),   x ≥ x_m
```

Foundational heavy-tail law. The "80-20 rule" corresponds
to `α = log 5 / log 4 ≈ 1.161`. Moments:

- `E[X] = α x_m / (α − 1)` for `α > 1`; otherwise infinite.
- `Var[X] = α x_m² / ((α − 1)² (α − 2))` for `α > 2`.

MLE has a clean closed form: `x̂_m = min(x)`,
`α̂ = n / Σ log(x_i / x_m)`.

## Files

- `python/pareto_distribution.py` — inverse-CDF sampling,
  MLE recovery to 3 decimals across three regimes.
  Illustrates the 80-20 rule: at `α ≈ 1.16`, the top 20 %
  of a 100 000-draw sample holds ~ 77 % of the total
  (theory 80 %; finite-sample tail sampling variance).
- `r/pareto_distribution.R` — `EnvStats::dpareto`,
  `actuar::dpareto1`, `poweRlaw::conpl` (R);
  `scipy.stats.pareto`, from-scratch (Python).

## Where else it appears in this repo

- `extreme-value-theory` — Pareto is a Fréchet-domain
  distribution; **generalised Pareto** is the excess-over-
  threshold model.
- `pareto-charts` — the FREQUENCY chart (Juran); named after
  Pareto but does not use his distribution directly.
- `cvar-expected-shortfall` — Pareto tails underlie
  catastrophic-loss modelling.
- `gini-lorenz` — Gini coefficient for Pareto has closed
  form `1 / (2α − 1)`.
- `small-world-scale-free`, `random-graph-models` —
  scale-free graph degrees follow discrete Pareto (Zipf).

## Assumptions & caveats

- **Threshold x_m** — sensitive to the smallest observation;
  in practice thresholds are chosen from a hill / MEEF plot,
  not just the sample min.
- **Continuous vs discrete** — Zipf's law is the discrete
  cousin (word / rank frequencies).
- **Test for power-law** — Clauset-Shalizi-Newman (2009)
  procedure; Kolmogorov-Smirnov distance + parametric bootstrap.
- **Lomax = shifted Pareto** — `Y = X − x_m` has support
  `[0, ∞)`; `numpy.random.pareto` returns this shifted
  version.
- **Tail-index estimation** — Hill estimator is standard;
  bias correction (Beirlant, Feller) improves it.

## Run

```
python techniques/pareto-distribution/python/pareto_distribution.py
Rscript techniques/pareto-distribution/r/pareto_distribution.R
```

**Refs:** Pareto, V. *Cours d'économie politique*, Rouge, Lausanne, 1897; Clauset, A., Shalizi, C.R. and Newman, M.E.J. "Power-law distributions in empirical data." *SIAM Rev.*, 51: 661-703, 2009.

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
