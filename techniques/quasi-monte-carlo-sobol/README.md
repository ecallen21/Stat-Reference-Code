# Quasi-Monte Carlo -- Sobol / Halton (Reference §45.9)

Niederreiter (1992); Sobol (1967); Owen (1997, scrambling). Replaces
IID uniform draws with **low-discrepancy sequences** that cover the
unit hypercube more evenly.

## Convergence

    Standard MC error  ~  O(1 / √N)
    QMC error          ~  O((log N)^d / N)      (bounded-variation f)

Randomised QMC (Owen scrambling) gives an **unbiased** estimator
whose variance still decays faster than 1/N.

## Files

- `python/quasi_monte_carlo_sobol.py` — MC vs scrambled Sobol on
  a monomial `∏ xᵢ²` and a radial `exp(−‖x − ½‖²)` from scratch
  (uses `scipy.stats.qmc.Sobol`). Demo: d=3, N=4096 gives Sobol RMSE
  158× smaller than MC; radial d=4, N=16384 gives Sobol variance
  ~5900× smaller than MC.
- `r/quasi_monte_carlo_sobol.R` — `randtoolbox::sobol / halton`,
  `qrng`, `spacefillr`, `fOptions` (R); `scipy.stats.qmc`,
  `SALib`, `torch.quasirandom.SobolEngine` (Python).

## When to use

- **Deterministic / low-dim integration** — Bayesian model evidence,
  option pricing, expected utility.
- **Sensitivity analysis** — Sobol indices for global sensitivity
  (SALib).
- **Space-filling experimental design** — Latin-hypercube or Sobol
  in DoE.
- **Faster ELBO estimation** — variational inference with QMC noise.

## When NOT to use

- **High-dim (d > ~40)** — the `log^d N` factor bites; scrambled QMC
  helps but doesn't beat plain MC by much.
- **Discontinuous integrands** — QMC's advantage assumes bounded
  variation; sharp jumps kill the rate.
- **Streaming / incremental estimation** — QMC needs the whole
  sequence up front.

## Assumptions & caveats

- **Sequence base powers of 2** for Sobol (best behaviour); `n =
  2ᵏ` from `random_base2`.
- **Scrambling** essential for reliable finite-sample SE estimation
  (Owen scrambling in scipy / spacefillr).
- **Independence claims** — QMC points are NOT iid; classical CLT
  formulas overestimate coverage.
- **Randomised replicates** — average over ≥10 independent scrambled
  sequences to estimate MC variance.

## Related in this repo

- `monte-carlo-simulation`, `importance-sampling`,
  `nonparametric-bootstrap` — MC family.
- `latin-hypercube-sampling` — space-filling cousin.
- `bridge-sampling-evidence`, `bayesian-optimization` — canonical
  Bayesian applications.
- `random-projections` — different "low discrepancy" spirit.

## Run

```
python techniques/quasi-monte-carlo-sobol/python/quasi_monte_carlo_sobol.py
Rscript techniques/quasi-monte-carlo-sobol/r/quasi_monte_carlo_sobol.R
```

**Refs:** Niederreiter, H. *Random Number Generation and Quasi-Monte Carlo Methods*, SIAM, 1992; Sobol, I.M. "On the distribution of points in a cube and the approximate evaluation of integrals." *USSR Comp. Math. Math. Phys.*, 7: 86-112, 1967; Owen, A. "Scrambled net variance for integrals of smooth functions." *Annals of Statistics*, 25(4): 1541-1562, 1997.

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
