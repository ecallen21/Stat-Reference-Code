# Inhomogeneous Poisson Process (Reference §47.96)

Cox (1955); Ogata (1981) for simulation. Point process on [0, T]
whose intensity depends on time:

    λ(t): ℝ₊ → ℝ₊,     E[N(a, b)] = ∫_a^b λ(t) dt.

Simulate via Ogata thinning: bound λ(t) ≤ M, propose t ~
Poisson(M), accept with probability λ(t) / M. MLE for a parametric
intensity by maximising

    log L(θ) = Σᵢ log λ(tᵢ; θ) − ∫₀ᵀ λ(t; θ) dt.

## Files

- `python/poisson_point_process_inhomog.py` — Ogata thinning
  simulator + Nelder-Mead MLE (`scipy.integrate.quad` for the
  compensator). Demo (T=100, `λ(t) = 0.4 + 0.3 sin(2π t/25) +
  2 exp(−t/20)`):
  - simulated N = 59 events (expected 79.7)
  - MLE α̂=0.36, β̂=+0.31, γ̂=1.64, τ̂=14.3 vs truth (0.40, 0.30, 2.0, 20).
- `r/poisson_point_process_inhomog.R` — `spatstat`,
  `PtProcess`, `NHPoisson` (R); `tick`, `lifelines`, custom (Python).

## When to use

- **Time-varying event rates** — call-centre arrivals with daily
  peaks, earthquake catalogs with mainshock-aftershock decay,
  ad-impression streams.
- **Piecewise-constant rate hazards** — extends to
  `piecewise-exponential-model` for survival with cuts.
- **Base-rate modelling** for Hawkes / cluster-process extensions.

## When NOT to use

- **Self-exciting or clustered events** — `hawkes-process` or
  Neyman-Scott is more appropriate.
- **Very long horizons with slow λ** — simulate on log-time or
  use time-change to homogeneous Poisson.
- **Marked events** — extend to marked-point processes.

## Assumptions & caveats

- **Ordered independent inter-arrivals** conditional on λ.
- **Bound M** in thinning must dominate λ over [0, T]; too tight →
  bias, too loose → many rejections.
- **Log-lik integral** — analytic for polynomial / exponential λ;
  quadrature otherwise.
- **Identifiability** — parametric families with correlated params
  can be poorly identified; use profile likelihood.

## Related in this repo

- `hawkes-process` — self-exciting extension.
- `piecewise-exponential-model`, `recurrent-events`,
  `parametric-survival` — related time-to-event models.
- `kernel-intensity-2d`, `ripleys-k-point-pattern`,
  `spatial-scan-cluster` — spatial-point-process cousins.
- `sequential-analysis`, `rare-event-control-charts` — related
  event-monitoring tooling.

## Run

```
python techniques/poisson-point-process-inhomog/python/poisson_point_process_inhomog.py
Rscript techniques/poisson-point-process-inhomog/r/poisson_point_process_inhomog.R
```

**Refs:** Cox, D.R. "Some statistical methods connected with series of events." *JRSS-B* 17(2): 129-164, 1955; Ogata, Y. "On Lewis' simulation method for point processes." *IEEE Trans Info Theory* 27(1): 23-31, 1981.

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
