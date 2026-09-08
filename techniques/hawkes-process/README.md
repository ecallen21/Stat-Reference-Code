# Hawkes Process (Reference §47.41)

Hawkes (1971). Self-exciting temporal point process. Conditional
intensity

    λ(t) = μ + Σ_{t_i < t} α exp(−β (t − t_i))

each past event boosts intensity by `α`, decaying at rate `β`.
Stationary iff branching ratio `n = α/β < 1`; then
`E[N(0, T)] = μT / (1 − n)`.

## Files

- `python/hawkes_process.py` — Ogata thinning simulation +
  exact log-likelihood MLE via Ozaki 1979 recursion. Demo
  (μ=0.5, α=0.6, β=1.5, T=500): simulated N=361 events,
  MLE (μ̂, α̂, β̂) = (0.53, 0.45, 1.65) vs truth (0.5, 0.6, 1.5).
- `r/hawkes_process.R` — `hawkes`, `PtProcess`, `evently`
  (R); `tick`, from-scratch (Python).

## When to use

- **Financial trades / order arrivals** — bursty clustering.
- **Earthquake aftershocks** — Hawkes originated here (Ogata 1988).
- **Social-media cascades** — retweet bursts.
- **Neural spike trains** — cross-excitation between neurons.
- **Epidemic modelling** — self-exciting branching.

## When NOT to use

- **Regular / periodic events** — use a renewal or Poisson process.
- **Long-range dependence** — exponential kernel is Markovian; use
  power-law kernel or Hawkes with slowly-decaying memory.
- **Non-stationary intensity** — `μ` is constant here; add trend
  or seasonality if needed.

## Assumptions & caveats

- **Branching ratio < 1** — else expected counts diverge.
- **Exp kernel** — most tractable; other choices (power-law) need
  Ogata thinning + numerical integrals.
- **Log-likelihood optimisation** — Nelder-Mead here; L-BFGS-B or
  EM (Veen-Schoenberg 2008) more robust in higher-dim.
- **Multivariate Hawkes** — extends to cross-excitation matrices;
  cost grows quadratically in dim.

## Related in this repo

- `poisson-regression`, `negative-binomial-regression` — count
  regression baselines.
- `recurrent-events` — AG/PWP models for cohort recurrent event data.
- `network-diffusion`, `temporal-networks` — event-driven graph
  processes.
- `state-space-kalman`, `state-space-models` — continuous-time latent
  dynamics analogues.

## Run

```
python techniques/hawkes-process/python/hawkes_process.py
Rscript techniques/hawkes-process/r/hawkes_process.R
```

**Refs:** Hawkes, A.G. "Spectra of some self-exciting and mutually exciting point processes." *Biometrika* 58(1): 83-90, 1971; Ogata, Y. "On Lewis' simulation method for point processes." *IEEE Trans Info Theory* 27(1): 23-31, 1981; Ozaki, T. "Maximum likelihood estimation of Hawkes' self-exciting point processes." *Ann Inst Stat Math* 31: 145-155, 1979.

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
