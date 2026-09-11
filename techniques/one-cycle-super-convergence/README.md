# One-Cycle / Super-Convergence (Reference §47.166)

Smith (2018). Uses a 1-cycle learning-rate schedule:

- **Phase 1**: lr rises linearly from lr_min → lr_max (~half run).
- **Phase 2**: lr falls linearly (or cosine) from lr_max → lr_min.
- **Phase 3 (optional)**: lr descends further, ~1-2 orders of
  magnitude lower.

Simultaneously, **momentum is reversed**: high in phases with low
lr, low when lr is at its peak. Enables 5-10× faster convergence
than fixed-lr schedules on CIFAR / ImageNet.

## Files

- `python/one_cycle_super_convergence.py` — SGD-momentum with
  fixed-lr vs 1-cycle schedules on a 50-D, 1500-sample
  classification. Includes the schedule shape at sample steps:
  - step 0    lr = 0.012
  - step 250  lr = 0.300 (peak)
  - step 500  lr = 0.000  (annealed).
  - On this convex problem 1-cycle matches a well-chosen fixed lr;
    its big wins are on non-convex nets where the ramp-up phase
    escapes sharp minima.
- `r/one_cycle_super_convergence.R` — pure-R schedule computation.

## When to use

- **Deep-net training with a compute budget** — 1-cycle often
  matches fixed-lr endpoint in a fraction of the epochs.
- **fastai / PyTorch practitioners** — this is the fastai default.
- **When lr tuning is expensive** — 1-cycle just needs a
  reasonable lr_max (find via a range test).

## When NOT to use

- **Very long training** where fixed lr / cosine annealing already
  work.
- **Convex objectives** — no super-convergence phenomenon.
- **Multi-task / continual training** where the anneal phase
  interferes with later tasks.

## Assumptions & caveats

- **lr_max** determined by a "range test" (Smith 2017): lr sweep
  showing where loss diverges.
- **Momentum inversion** (0.85 → 0.95 or similar) is part of the
  original recipe.
- **Total-run schedule** — the whole training run is the cycle,
  not per-epoch cycles.
- **Fixed budget** — you must commit to a run length up front.

## Related in this repo

- `lr-schedules`, `adam-optimizer`, `gradient-clipping` — training
  toolbox.
- `lookahead-optimizer`, `rectified-adam-radam` — alternative
  variance-reduction paths.
- `stochastic-weight-averaging-swa` — end-of-training averaging
  companion.

## Run

```
python techniques/one-cycle-super-convergence/python/one_cycle_super_convergence.py
Rscript techniques/one-cycle-super-convergence/r/one_cycle_super_convergence.R
```

**Refs:** Smith, L. N. "Super-convergence: Very fast training of neural networks using large learning rates." *arXiv:1708.07120*, 2018; Smith, L. N. "Cyclical learning rates for training neural networks." *WACV*, 2017.

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
