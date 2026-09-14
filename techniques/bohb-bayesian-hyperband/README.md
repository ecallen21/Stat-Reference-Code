# BOHB — Bayesian Optimisation + Hyperband (Reference §47.280)

Falkner, Klein & Hutter (2018). Combines the parallel budget-
efficient EXPLORATION of Hyperband with the sample-efficient
EXPLOITATION of a TPE surrogate model:

- Hyperband outer loop: brackets of successive halving.
- Inside each bracket, ask the TPE model for informed picks
  instead of pure random sampling.

Empirically stronger than either alone; core of Ray Tune's
BOHB scheduler and AutoML competition winners.

## Files

- `python/bohb_bayesian_hyperband.py` — Simplified 1-D BOHB.
  Toy: bimodal function with true min at x=0.7 and a decoy at
  x=0.15. R=27, η=3. BOHB finds x≈0.685 (loss ≈ −0.04) using
  71 evaluations; plain Hyperband at same budget lands at
  x≈0.81 (settled onto local structure) — TPE-guided
  sampling steered proposals toward the true mode faster.
- `r/bohb_bayesian_hyperband.R` — reticulate + Ray Tune
  BOHBScheduler, mlr3hyperband + mlr3mbo (R); hpbandster,
  Ray Tune, SMAC3, from-scratch (Python).

## When to use

- **Modern AutoML pipelines** — BOHB is a strong practical
  default for hyperparameter search.
- **Distributed workers + partial-budget signal** — full
  Hyperband + informed proposals.
- **After a few random-search warmup trials** — BOHB benefits
  from history.

## When NOT to use

- **Very small search budget** — the TPE overhead is wasted
  when only 10-20 evaluations are affordable.
- **Non-continuous / non-tree HPs** — BOHB's TPE assumes a
  tree-structured space; use SMAC for arbitrary categorical /
  conditional.
- **Extreme parallelism (100s workers)** — ASHA + random or
  ASHA + BO with re-sampling may scale better.

## Assumptions & caveats

- **Budget rungs** — same geometric progression as Hyperband
  (r, rη, rη²,...).
- **TPE γ, bandwidth** — same tuning knobs as pure TPE.
- **Cold-start** — BOHB needs a minimum number of full-budget
  observations for the TPE surrogate to matter; keep at least
  one "s=0" bracket.
- **Reporting** — report the config that achieved the best
  score at the HIGHEST rung reached; not the current-best
  overall.

## Related in this repo

- `hyperband-multi-fidelity` — outer bracket loop.
- `successive-halving-asha` — inner promotion loop.
- `tpe-tree-parzen-estimator` — the SMBO sampler used inside.
- `bayesian-optimization` — non-multi-fidelity BO.
- `population-based-training` — evolutionary alternative.

## Run

```
python techniques/bohb-bayesian-hyperband/python/bohb_bayesian_hyperband.py
Rscript techniques/bohb-bayesian-hyperband/r/bohb_bayesian_hyperband.R
```

**Refs:** Falkner, S., Klein, A. and Hutter, F. "BOHB: Robust and efficient hyperparameter optimization at scale." In *ICML*, pp. 1437-1446, 2018.

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
