# Antithetic + Control Variates (Reference §45.10)

Hammersley & Morton (1956, antithetic); classical Monte-Carlo
variance reduction.

## Antithetic

For `μ = E[f(U)]` with `U ~ U(0, 1)`, pair each `Uᵢ` with `1 − Uᵢ`:

    μ̂_a = (1/n) Σ (f(Uᵢ) + f(1 − Uᵢ)) / 2
    Var(μ̂_a) < Var(μ̂_MC)  when Cov(f(U), f(1 − U)) < 0

Cheapest possible variance reduction when `f` is monotone in `U`.

## Control variate

Choose `g(U)` with known `E[g] = α`:

    μ̂_c = f̄ − c · (ḡ − α),   optimal c* = Cov(f, g) / Var(g)
    Variance reduction  = 1 − ρ²(f, g)

## Files

- `python/antithetic_control_variates.py` — plain MC, antithetic
  MC, and CV estimators (with `g(u) = u`) for `E[exp(U)] = e − 1`
  from scratch. Demo (n=1000, 500 reps): antithetic cuts variance
  28×, control variate 54× vs plain MC.
- `r/antithetic_control_variates.R` — no dedicated CRAN package;
  base-R one-liners.

## When to use

- **Any smooth monotone integrand** — antithetic gets a free
  variance cut.
- **A cheap correlated proxy exists** — CV shines when `ρ(f, g)` is
  high.
- **Financial / stochastic-simulation** — control variates are
  standard for option pricing.

## When NOT to use

- **Highly non-monotone `f`** — antithetic can INCREASE variance if
  `Cov(f, 1−f) > 0`.
- **No useful control variate** — CV needs an analytically known
  `E[g]`.
- **Deterministic integrals with QMC** — Sobol / Halton can beat MC
  more decisively.

## Assumptions & caveats

- **Antithetic assumes uniform inputs** — for other distributions,
  invert to uniform first (`Φ⁻¹`, `F⁻¹`).
- **CV bias** if `c*` is estimated from the same MC run — usually
  negligible; use split-sample or jackknife if worried.
- **Multiple control variates** — the OLS coefficient
  `c* = Cov(g, g)⁻¹ Cov(g, f)` generalises.
- **Combine with common random numbers** — pair antithetic + CRN
  across scenarios for further variance cuts.

## Related in this repo

- `monte-carlo-simulation`, `importance-sampling`, `quasi-monte-carlo-sobol`
  — MC family.
- `cuped-variance-reduction`, `prognostic-score-covariate-adjustment`
  — experimental-analysis analogues.
- `bridge-sampling-evidence` — advanced ratio estimator with
  bridge-derived variance cuts.

## Run

```
python techniques/antithetic-control-variates/python/antithetic_control_variates.py
Rscript techniques/antithetic-control-variates/r/antithetic_control_variates.R
```

**Refs:** Hammersley, J.M. & Morton, K.W. "A new Monte Carlo technique: antithetic variates." *Mathematical Proceedings of the Cambridge Philosophical Society*, 52(3): 449-475, 1956; Glasserman, P. *Monte Carlo Methods in Financial Engineering*, Springer, 2003.

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
