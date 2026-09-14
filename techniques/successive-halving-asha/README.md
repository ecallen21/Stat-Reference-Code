# Successive Halving / ASHA (Reference §47.276)

Karnin, Koren & Somekh (2013, SH); Li, Jamieson et al (2020,
ASHA). Repeatedly evaluate `n` configs at budget `r`, keep the
top 1/η, promote survivors to budget `r · η`:

- **Successive Halving (synchronous)**: wait for all `n`
  before halving.
- **ASHA (asynchronous)**: promote whenever η pass a rung.

ASHA scales linearly across workers; single-worker SH is a
special case. Used as the inner loop of Hyperband and BOHB.

## Files

- `python/successive_halving_asha.py` — Both sync SH and a
  simplified ASHA scheduler. Toy: (lr, hidden) search. SH
  with n=27, r=1, η=3 goes 27 → 9 → 3 → 1 with budget
  1 → 3 → 9 → 27. ASHA over 40 configs promotes eagerly
  through rungs 1 → 3 → 9 → 27. Both converge to near
  (lr=0.01, hidden=64).
- `r/successive_halving_asha.R` — `mlr3hyperband`,
  reticulate + Ray Tune (R); Ray Tune ASHAScheduler, Optuna
  SuccessiveHalvingPruner, from-scratch (Python).

## When to use

- **Distributed HP search** — ASHA is the practical default
  when workers are available.
- **Inside Hyperband / BOHB** — SH is the inner loop.
- **Cheap-early scoring** — validation loss after few epochs
  is a reasonable ranking signal.

## When NOT to use

- **Small search budget** — SH's overhead is high relative
  to the gain when N is small.
- **Single worker with expensive early stop** — the promotion
  latency of sync SH is fine; ASHA's async ceases to help.
- **Correlated noise** — if low-budget scores are highly
  noisy, use averaged / repeated evaluations.

## Assumptions & caveats

- **η choice** — 2-4 is standard; η=3 is the canonical
  Hyperband default.
- **Rung structure** — geometric rungs (r, rη, rη²,...)
  match the SH algorithm; deviating hurts guarantees.
- **Discarding rate** — SH discards (η−1)/η per rung; be
  aware you may drop a slow-starter that ends best.
- **Async safety** — ASHA's async promotion means the "current
  best" can change; report the config that WAS best at the
  highest rung reached.

## Related in this repo

- `hyperband-multi-fidelity` — outer bracket loop wrapping SH.
- `bohb-bayesian-hyperband` — SH + TPE sampling.
- `bayesian-optimization` — non-multi-fidelity alternative.

## Run

```
python techniques/successive-halving-asha/python/successive_halving_asha.py
Rscript techniques/successive-halving-asha/r/successive_halving_asha.R
```

**Refs:** Karnin, Z., Koren, T. and Somekh, O. "Almost optimal exploration in multi-armed bandits." In *ICML*, pp. 1238-1246, 2013; Li, L. et al. "A system for massively parallel hyperparameter tuning." In *MLSys*, 2020.

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
